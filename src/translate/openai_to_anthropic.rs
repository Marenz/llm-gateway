use crate::types::{
    AnthropicContentBlock, AnthropicImageSource, AnthropicMessage, AnthropicMessageContent,
    AnthropicNamedToolChoice, AnthropicRequest, AnthropicSystem, AnthropicSystemMessage,
    AnthropicTool, AnthropicToolChoice, AnthropicToolChoiceMode, OpenAIContentPart, OpenAIMessage,
    OpenAIMessageContent, OpenAIStop, OpenAIToolChoice,
};

/// Required system prompt prefix for Anthropic OAuth tokens to unlock non-haiku models.
const OAUTH_REQUIRED_SYSTEM: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

pub fn translate_request(
    req: &crate::types::OpenAIRequest,
) -> Result<crate::types::AnthropicRequest, String> {
    let cache_controls = req
        .messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            message
                .cache_control
                .as_ref()
                .map(|marker| validate_cache_control(marker, index))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let explicit_breakpoints = cache_controls.iter().any(Option::is_some);
    let (system, mut placed_breakpoints) =
        collect_system_prompt(&req.messages, &cache_controls, explicit_breakpoints)?;
    let mut messages = Vec::new();
    let mut message_breakpoints: Vec<Option<(usize, serde_json::Value)>> = Vec::new();

    for (source_index, message) in req.messages.iter().enumerate() {
        let cache_control = cache_controls[source_index].clone();
        match message.role {
            crate::types::OpenAIRole::System => {}
            crate::types::OpenAIRole::User => {
                let index = push_message(
                    &mut messages,
                    AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicMessageContent::Blocks(translate_user_content(
                            &message.content,
                        )),
                    },
                );
                remember_breakpoint(&mut message_breakpoints, index, source_index, cache_control)?;
            }
            crate::types::OpenAIRole::Assistant => {
                let index = push_message(&mut messages, translate_assistant_message(message));
                remember_breakpoint(&mut message_breakpoints, index, source_index, cache_control)?;
            }
            crate::types::OpenAIRole::Tool => {
                if let Some(tool_call_id) = &message.tool_call_id {
                    let index = push_message(
                        &mut messages,
                        AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicMessageContent::Blocks(vec![
                                AnthropicContentBlock::ToolResult {
                                    tool_use_id: tool_call_id.clone(),
                                    content: serde_json::Value::String(extract_text(
                                        &message.content,
                                    )),
                                    cache_control: None,
                                },
                            ]),
                        },
                    );
                    remember_breakpoint(
                        &mut message_breakpoints,
                        index,
                        source_index,
                        cache_control,
                    )?;
                } else if cache_control.is_some() {
                    return Err(format!(
                        "messages[{source_index}].cache_control cannot be represented: tool message has no tool_call_id"
                    ));
                }
            }
        }
    }

    if explicit_breakpoints {
        for (message_index, breakpoint) in message_breakpoints.into_iter().enumerate() {
            if let Some((source_index, marker)) = breakpoint {
                let message = messages.get_mut(message_index).ok_or_else(|| {
                    format!(
                        "messages[{source_index}].cache_control could not be mapped after role merging"
                    )
                })?;
                set_final_block_cache_control(message, marker).map_err(|reason| {
                    format!(
                        "messages[{source_index}].cache_control cannot be represented: {reason}"
                    )
                })?;
                placed_breakpoints += 1;
            }
        }
        if placed_breakpoints > 4 {
            return Err(format!(
                "Anthropic supports at most 4 cache breakpoints, but this request produces {placed_breakpoints} after role merging"
            ));
        }
    } else {
        // Preserve the legacy bounded heuristic when the client did not choose
        // explicit boundaries: one system marker and up to three message turns.
        apply_cache_breakpoints(&mut messages);
    }

    let mut tools = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|tool| AnthropicTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                input_schema: tool.function.parameters.clone(),
            })
            .collect::<Vec<_>>()
    });

    let tool_choice = match req.tool_choice.as_ref() {
        Some(OpenAIToolChoice::Mode(mode)) => match mode.as_str() {
            "auto" => Some(AnthropicToolChoice::Mode(AnthropicToolChoiceMode {
                kind: "auto".to_string(),
            })),
            "required" => Some(AnthropicToolChoice::Mode(AnthropicToolChoiceMode {
                kind: "any".to_string(),
            })),
            "none" => {
                tools = None;
                None
            }
            _ => None,
        },
        Some(OpenAIToolChoice::Tool(choice)) => {
            Some(AnthropicToolChoice::Named(AnthropicNamedToolChoice {
                kind: "tool".to_string(),
                name: choice.function.name.clone(),
            }))
        }
        None => None,
    };

    Ok(AnthropicRequest {
        model: req.model.clone(),
        max_tokens: req.max_tokens.unwrap_or(4096),
        messages,
        system,
        tools,
        tool_choice,
        stream: req.stream,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: None,
        stop_sequences: translate_stop(req.stop.as_ref()),
        metadata: None,
        thinking: translate_thinking(req.reasoning_effort.as_deref()),
    })
}

fn collect_system_prompt(
    messages: &[OpenAIMessage],
    cache_controls: &[Option<serde_json::Value>],
    explicit_breakpoints: bool,
) -> Result<(Option<AnthropicSystem>, usize), String> {
    // Always prepend the required OAuth system prompt so non-haiku models work.
    let mut blocks = vec![AnthropicSystemMessage {
        kind: "text".to_string(),
        text: OAUTH_REQUIRED_SYSTEM.to_string(),
        cache_control: None,
    }];

    let mut placed_breakpoints = 0;
    for (index, message) in messages
        .iter()
        .enumerate()
        .filter(|(_, message)| matches!(message.role, crate::types::OpenAIRole::System))
    {
        let text = extract_text(&message.content);
        let cache_control = cache_controls[index].clone();
        if !text.trim().is_empty() {
            placed_breakpoints += usize::from(cache_control.is_some());
            blocks.push(AnthropicSystemMessage {
                kind: "text".to_string(),
                text,
                cache_control,
            });
        } else if cache_control.is_some() {
            return Err(format!(
                "messages[{index}].cache_control cannot be represented: system message has no non-empty text"
            ));
        }
    }

    if !explicit_breakpoints {
        // The required OAuth prefix guarantees a non-empty final system block.
        if let Some(last) = blocks.last_mut() {
            last.cache_control = Some(serde_json::json!({"type": "ephemeral"}));
        }
    }

    Ok((Some(AnthropicSystem::Messages(blocks)), placed_breakpoints))
}

fn validate_cache_control(
    marker: &serde_json::Value,
    message_index: usize,
) -> Result<serde_json::Value, String> {
    let Some(object) = marker.as_object() else {
        return Err(format!(
            "messages[{message_index}].cache_control must be an object"
        ));
    };
    if object.get("type").and_then(serde_json::Value::as_str) != Some("ephemeral") {
        return Err(format!(
            "messages[{message_index}].cache_control.type must be \"ephemeral\""
        ));
    }
    if object.keys().any(|key| key != "type" && key != "ttl") {
        return Err(format!(
            "messages[{message_index}].cache_control supports only type and ttl"
        ));
    }
    if let Some(ttl) = object.get("ttl") {
        if !matches!(ttl.as_str(), Some("5m" | "1h")) {
            return Err(format!(
                "messages[{message_index}].cache_control.ttl must be \"5m\" or \"1h\""
            ));
        }
    }
    Ok(marker.clone())
}

fn remember_breakpoint(
    breakpoints: &mut Vec<Option<(usize, serde_json::Value)>>,
    message_index: usize,
    source_index: usize,
    marker: Option<serde_json::Value>,
) -> Result<(), String> {
    if breakpoints.len() <= message_index {
        breakpoints.resize(message_index + 1, None);
    }
    if let Some(marker) = marker {
        if let Some((previous_source, _)) = &breakpoints[message_index] {
            return Err(format!(
                "messages[{previous_source}] and messages[{source_index}] request distinct cache boundaries that collapse into one Anthropic message after same-role merging"
            ));
        }
        breakpoints[message_index] = Some((source_index, marker));
    }
    Ok(())
}

fn set_final_block_cache_control(
    message: &mut AnthropicMessage,
    marker: serde_json::Value,
) -> Result<(), &'static str> {
    match &mut message.content {
        AnthropicMessageContent::Text(text) => {
            if text.trim().is_empty() {
                return Err("final text block is empty");
            }
            let text = std::mem::take(text);
            message.content = AnthropicMessageContent::Blocks(vec![AnthropicContentBlock::Text {
                text,
                cache_control: Some(marker),
            }]);
            Ok(())
        }
        AnthropicMessageContent::Blocks(blocks) => match blocks.last_mut() {
            Some(AnthropicContentBlock::Text {
                text,
                cache_control,
            }) if !text.trim().is_empty() => {
                *cache_control = Some(marker);
                Ok(())
            }
            Some(AnthropicContentBlock::Image { cache_control, .. })
            | Some(AnthropicContentBlock::ToolUse { cache_control, .. })
            | Some(AnthropicContentBlock::ToolResult { cache_control, .. }) => {
                *cache_control = Some(marker);
                Ok(())
            }
            Some(AnthropicContentBlock::Text { .. }) => Err("final text block is empty"),
            Some(AnthropicContentBlock::Thinking { .. }) => {
                Err("final thinking block is not cacheable")
            }
            None => Err("message has no content blocks"),
        },
    }
}

/// Apply cache_control breakpoints to the last 3 messages (using up remaining 3 of 4 allowed).
/// Attaches to the last content block of each targeted message.
fn apply_cache_breakpoints(messages: &mut Vec<AnthropicMessage>) {
    let cache_marker = serde_json::json!({"type": "ephemeral"});
    let n = messages.len();
    let targets: Vec<usize> = (0..3).filter_map(|i| n.checked_sub(i + 1)).collect();

    for idx in targets {
        if let Some(msg) = messages.get_mut(idx) {
            match &mut msg.content {
                AnthropicMessageContent::Blocks(blocks) => {
                    if let Some(last_block) = blocks.last_mut() {
                        match last_block {
                            // Anthropic rejects cache_control on empty text blocks
                            // ("cache_control cannot be set for empty text blocks").
                            // Skip the breakpoint when the block text is blank.
                            AnthropicContentBlock::Text {
                                text,
                                cache_control,
                            } => {
                                if !text.trim().is_empty() {
                                    *cache_control = Some(cache_marker.clone());
                                }
                            }
                            AnthropicContentBlock::Image { cache_control, .. }
                            | AnthropicContentBlock::ToolUse { cache_control, .. }
                            | AnthropicContentBlock::ToolResult { cache_control, .. } => {
                                *cache_control = Some(cache_marker.clone());
                            }
                            _ => {}
                        }
                    }
                }
                AnthropicMessageContent::Text(t) => {
                    let text = t.clone();
                    // Only attach a cache breakpoint to non-empty text; an empty
                    // string would trigger Anthropic's empty-text-block error.
                    let cache_control = if text.trim().is_empty() {
                        None
                    } else {
                        Some(cache_marker.clone())
                    };
                    msg.content =
                        AnthropicMessageContent::Blocks(vec![AnthropicContentBlock::Text {
                            text,
                            cache_control,
                        }]);
                }
            }
        }
    }
}

fn translate_assistant_message(message: &OpenAIMessage) -> AnthropicMessage {
    let mut blocks = Vec::new();
    let text = extract_text(&message.content);

    if !text.is_empty() {
        blocks.push(AnthropicContentBlock::Text {
            text,
            cache_control: None,
        });
    }

    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            let input = serde_json::from_str(&tool_call.function.arguments).unwrap_or_else(|_| {
                serde_json::Value::String(tool_call.function.arguments.clone())
            });
            blocks.push(AnthropicContentBlock::ToolUse {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                input,
                cache_control: None,
            });
        }
    }

    AnthropicMessage {
        role: "assistant".to_string(),
        content: match blocks.as_slice() {
            [] => AnthropicMessageContent::Text(String::new()),
            [
                AnthropicContentBlock::Text {
                    text,
                    cache_control: None,
                },
            ] => AnthropicMessageContent::Text(text.clone()),
            _ => AnthropicMessageContent::Blocks(blocks),
        },
    }
}

fn translate_user_content(content: &OpenAIMessageContent) -> Vec<AnthropicContentBlock> {
    match content {
        OpenAIMessageContent::Text(text) => vec![AnthropicContentBlock::Text {
            text: text.clone(),
            cache_control: None,
        }],
        OpenAIMessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                OpenAIContentPart::Text { text, .. } => AnthropicContentBlock::Text {
                    text: text.clone(),
                    cache_control: None,
                },
                OpenAIContentPart::ImageUrl { image_url } => AnthropicContentBlock::Image {
                    source: translate_image_source(&image_url.url),
                    cache_control: None,
                },
            })
            .collect(),
        OpenAIMessageContent::Null => Vec::new(),
    }
}

fn translate_image_source(url: &str) -> AnthropicImageSource {
    if let Some(data_url) = url.strip_prefix("data:") {
        if let Some((media_type, data)) = data_url.split_once(";base64,") {
            return AnthropicImageSource::Base64 {
                media_type: media_type.to_string(),
                data: data.to_string(),
            };
        }
    }

    AnthropicImageSource::Url {
        url: url.to_string(),
    }
}

fn extract_text(content: &OpenAIMessageContent) -> String {
    match content {
        OpenAIMessageContent::Text(text) => text.clone(),
        OpenAIMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                crate::types::OpenAIContentPart::Text { text, .. } => Some(text.as_str()),
                crate::types::OpenAIContentPart::ImageUrl { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
        OpenAIMessageContent::Null => String::new(),
    }
}

fn translate_stop(stop: Option<&OpenAIStop>) -> Option<Vec<String>> {
    match stop {
        Some(OpenAIStop::Single(value)) => Some(vec![value.clone()]),
        Some(OpenAIStop::Multiple(values)) if !values.is_empty() => Some(values.clone()),
        _ => None,
    }
}

fn translate_thinking(reasoning_effort: Option<&str>) -> Option<serde_json::Value> {
    let budget_tokens = match reasoning_effort {
        Some("low") => Some(1024),
        Some("medium") => Some(2048),
        Some("high") => Some(4096),
        _ => None,
    };

    budget_tokens.map(|budget_tokens| {
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": budget_tokens,
        })
    })
}

fn push_message(messages: &mut Vec<AnthropicMessage>, message: AnthropicMessage) -> usize {
    if let Some(last) = messages.last_mut() {
        if same_role(&last.role, &message.role) {
            last.content = merge_content(
                std::mem::replace(
                    &mut last.content,
                    AnthropicMessageContent::Text(String::new()),
                ),
                message.content,
            );
            return messages.len() - 1;
        }
    }

    messages.push(message);
    messages.len() - 1
}

fn same_role(left: &str, right: &str) -> bool {
    left == right
}

fn merge_content(
    left: AnthropicMessageContent,
    right: AnthropicMessageContent,
) -> AnthropicMessageContent {
    match (left, right) {
        (AnthropicMessageContent::Text(left), AnthropicMessageContent::Text(right)) => {
            match (left.is_empty(), right.is_empty()) {
                (true, _) => AnthropicMessageContent::Text(right),
                (_, true) => AnthropicMessageContent::Text(left),
                _ => AnthropicMessageContent::Text(format!("{left}\n\n{right}")),
            }
        }
        (left, right) => AnthropicMessageContent::Blocks(
            [content_into_blocks(left), content_into_blocks(right)]
                .into_iter()
                .flatten()
                .collect(),
        ),
    }
}

fn content_into_blocks(content: AnthropicMessageContent) -> Vec<AnthropicContentBlock> {
    match content {
        AnthropicMessageContent::Text(text) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![AnthropicContentBlock::Text {
                    text,
                    cache_control: None,
                }]
            }
        }
        AnthropicMessageContent::Blocks(blocks) => blocks,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::translate_request;
    use crate::types::OpenAIRequest;

    #[test]
    fn translates_base64_image_parts_for_anthropic() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="}
                    }
                ]
            }]
        }))
        .unwrap();

        let translated = serde_json::to_value(translate_request(&request).unwrap()).unwrap();

        assert_eq!(
            translated["messages"][0]["content"],
            json!([
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aGVsbG8="
                    },
                    "cache_control": {"type": "ephemeral"}
                }
            ])
        );
    }

    #[test]
    fn translates_remote_image_urls_for_anthropic() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.webp", "detail": "high"}
                }]
            }]
        }))
        .unwrap();

        let translated = serde_json::to_value(translate_request(&request).unwrap()).unwrap();

        assert_eq!(
            translated["messages"][0]["content"][0],
            json!({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.webp"
                },
                "cache_control": {"type": "ephemeral"}
            })
        );
    }

    #[test]
    fn explicit_breakpoints_replace_heuristic_and_follow_role_merging() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "system",
                    "content": "stable prefix",
                    "cache_control": {"type": "ephemeral"}
                },
                {"role": "system", "content": "volatile tail"},
                {
                    "role": "user",
                    "content": "first adjacent user block",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                },
                {"role": "user", "content": "second adjacent user block"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"}
                    }],
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }))
        .unwrap();

        let translated = serde_json::to_value(translate_request(&request).unwrap()).unwrap();

        assert_eq!(
            translated["system"][1]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert!(translated["system"][0].get("cache_control").is_none());
        assert!(translated["system"][2].get("cache_control").is_none());
        assert!(
            translated["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            translated["messages"][0]["content"][1]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert_eq!(
            translated["messages"][1]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[test]
    fn explicit_image_boundary_preserves_multipart_content() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-6",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                ],
                "cache_control": {"type": "ephemeral"}
            }]
        }))
        .unwrap();

        let translated = serde_json::to_value(translate_request(&request).unwrap()).unwrap();
        let blocks = translated["messages"][0]["content"].as_array().unwrap();

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn absent_explicit_breakpoints_keeps_bounded_heuristic() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"}
            ]
        }))
        .unwrap();

        let translated = serde_json::to_value(translate_request(&request).unwrap()).unwrap();

        assert_eq!(
            translated["system"][1]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert!(
            translated["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        for index in 1..=3 {
            assert_eq!(
                translated["messages"][index]["content"][0]["cache_control"],
                json!({"type": "ephemeral"})
            );
        }
    }

    #[test]
    fn explicit_empty_boundary_is_rejected_without_heuristic_fallback() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-6",
            "messages": [{
                "role": "user",
                "content": "   ",
                "cache_control": {"type": "ephemeral"}
            }]
        }))
        .unwrap();

        let error = translate_request(&request).unwrap_err();

        assert!(error.contains("messages[0].cache_control cannot be represented"));
        assert!(error.contains("empty"));
    }

    #[test]
    fn more_than_four_explicit_breakpoints_is_rejected() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "system", "content": "s", "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": "u1", "cache_control": {"type": "ephemeral"}},
                {"role": "assistant", "content": "a1", "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": "u2", "cache_control": {"type": "ephemeral"}},
                {"role": "assistant", "content": "a2", "cache_control": {"type": "ephemeral"}}
            ]
        }))
        .unwrap();

        let error = translate_request(&request).unwrap_err();

        assert!(error.contains("at most 4 cache breakpoints"));
        assert!(error.contains("produces 5"));
    }

    #[test]
    fn multiple_boundaries_collapsed_by_role_merging_are_rejected() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "one", "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": "two", "cache_control": {"type": "ephemeral"}}
            ]
        }))
        .unwrap();

        let error = translate_request(&request).unwrap_err();

        assert!(error.contains("distinct cache boundaries"));
        assert!(error.contains("same-role merging"));
    }

    #[test]
    fn malformed_cache_control_is_rejected() {
        let request: OpenAIRequest = serde_json::from_value(json!({
            "model": "claude-opus-4-6",
            "messages": [{
                "role": "user",
                "content": "hello",
                "cache_control": {"type": "persistent"}
            }]
        }))
        .unwrap();

        let error = translate_request(&request).unwrap_err();

        assert!(error.contains("cache_control.type must be \"ephemeral\""));
    }
}
