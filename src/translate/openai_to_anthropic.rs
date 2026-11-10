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
    validate_ttl_order(&req.messages, &cache_controls)?;
    let (system, mut placed_breakpoints) =
        collect_system_prompt(&req.messages, &cache_controls, explicit_breakpoints)?;
    let mut messages = Vec::new();

    for (source_index, message) in req.messages.iter().enumerate() {
        let cache_control = cache_controls[source_index].clone();
        let translated = match message.role {
            crate::types::OpenAIRole::System => continue,
            crate::types::OpenAIRole::User => AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Blocks(translate_user_content(&message.content)),
            },
            crate::types::OpenAIRole::Assistant => translate_assistant_message(message),
            crate::types::OpenAIRole::Tool => {
                let Some(tool_call_id) = &message.tool_call_id else {
                    if cache_control.is_some() {
                        return Err(format!(
                            "messages[{source_index}].cache_control cannot be represented: tool message has no tool_call_id"
                        ));
                    }
                    continue;
                };
                AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::ToolResult {
                            tool_use_id: tool_call_id.clone(),
                            content: serde_json::Value::String(extract_text(&message.content)),
                            cache_control: None,
                        },
                    ]),
                }
            }
        };

        // Same-role neighbours merge into one Anthropic message. Right after
        // the merge, the content of this source message is the tail of that
        // message, so its boundary goes on the current final block and stays
        // there when later messages are appended behind it.
        let index = push_message(&mut messages, translated);
        if let Some(marker) = cache_control {
            let newly_placed = set_final_block_cache_control(&mut messages[index], marker)
                .map_err(|reason| {
                    format!("messages[{source_index}].cache_control cannot be represented: {reason}")
                })?;
            placed_breakpoints += usize::from(newly_placed);
        }
    }

    if explicit_breakpoints {
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

/// Anthropic requires boundaries with the longer lifetime to come first in the
/// prompt, which is ordered system text, then messages. A marker without `ttl`
/// has the 5-minute lifetime.
fn validate_ttl_order(
    messages: &[OpenAIMessage],
    cache_controls: &[Option<serde_json::Value>],
) -> Result<(), String> {
    let is_system = |index: &usize| matches!(messages[*index].role, crate::types::OpenAIRole::System);
    let prompt_order = (0..messages.len())
        .filter(is_system)
        .chain((0..messages.len()).filter(|index| !is_system(index)));

    let mut first_five_minute = None;
    for index in prompt_order {
        let Some(marker) = &cache_controls[index] else {
            continue;
        };
        if marker.get("ttl").and_then(serde_json::Value::as_str) == Some("1h") {
            if let Some(earlier) = first_five_minute {
                return Err(format!(
                    "messages[{index}].cache_control.ttl \"1h\" comes after the 5-minute boundary of messages[{earlier}]; Anthropic requires 1-hour boundaries before 5-minute ones (system text precedes all other messages)"
                ));
            }
        } else if first_five_minute.is_none() {
            first_five_minute = Some(index);
        }
    }
    Ok(())
}

/// Put `marker` on the final content block of `message`. Returns whether this
/// added a breakpoint: a message that contributed no block of its own shares
/// the block, and so the boundary, of the message merged before it.
fn set_final_block_cache_control(
    message: &mut AnthropicMessage,
    marker: serde_json::Value,
) -> Result<bool, &'static str> {
    let slot = match &mut message.content {
        AnthropicMessageContent::Text(text) => {
            if text.trim().is_empty() {
                return Err("final text block is empty");
            }
            let text = std::mem::take(text);
            message.content = AnthropicMessageContent::Blocks(vec![AnthropicContentBlock::Text {
                text,
                cache_control: Some(marker),
            }]);
            return Ok(true);
        }
        AnthropicMessageContent::Blocks(blocks) => match blocks.last_mut() {
            Some(AnthropicContentBlock::Text {
                text,
                cache_control,
            }) if !text.trim().is_empty() => cache_control,
            Some(AnthropicContentBlock::Image { cache_control, .. })
            | Some(AnthropicContentBlock::ToolUse { cache_control, .. })
            | Some(AnthropicContentBlock::ToolResult { cache_control, .. }) => cache_control,
            Some(AnthropicContentBlock::Text { .. }) => return Err("final text block is empty"),
            Some(AnthropicContentBlock::Thinking { .. }) => {
                return Err("final thinking block is not cacheable");
            }
            None => return Err("message has no content blocks"),
        },
    };
    match slot {
        None => {
            *slot = Some(marker);
            Ok(true)
        }
        Some(existing) if *existing == marker => Ok(false),
        Some(_) => Err("the block already carries a boundary with a different ttl"),
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

    fn translated(request: serde_json::Value) -> serde_json::Value {
        let request: OpenAIRequest = serde_json::from_value(request).unwrap();
        serde_json::to_value(translate_request(&request).unwrap()).unwrap()
    }

    fn translation_error(request: serde_json::Value) -> String {
        let request: OpenAIRequest = serde_json::from_value(request).unwrap();
        translate_request(&request).unwrap_err()
    }

    /// The `cache_control` of every content block, message by message.
    fn markers(translated: &serde_json::Value) -> Vec<Vec<serde_json::Value>> {
        translated["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| {
                message["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|block| block.get("cache_control").cloned().unwrap_or(json!(null)))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn explicit_breakpoints_replace_heuristic_and_stay_on_their_source_message() {
        let translated = translated(json!({
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "system",
                    "content": "stable prefix",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
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
        }));

        assert_eq!(
            translated["system"][1]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert!(translated["system"][0].get("cache_control").is_none());
        assert!(translated["system"][2].get("cache_control").is_none());
        // The boundary stays on the first user message's block; the second
        // user message is merged behind it and remains outside the prefix.
        assert_eq!(
            markers(&translated),
            vec![
                vec![json!({"type": "ephemeral", "ttl": "1h"}), json!(null)],
                vec![json!({"type": "ephemeral"})],
            ]
        );
        assert_eq!(
            translated["messages"][0]["content"][1]["text"],
            "second adjacent user block"
        );
    }

    #[test]
    fn boundaries_on_a_tool_result_and_the_following_user_message_are_both_kept() {
        let translated = translated(json!({
            "model": "claude-opus-5",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call-1", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "call-1", "content": "result",
                 "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": "follow-up",
                 "cache_control": {"type": "ephemeral"}}
            ]
        }));

        assert_eq!(translated["messages"].as_array().unwrap().len(), 3);
        assert_eq!(
            translated["messages"][2]["content"],
            json!([
                {"type": "tool_result", "tool_use_id": "call-1", "content": "result",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "follow-up", "cache_control": {"type": "ephemeral"}}
            ])
        );
    }

    #[test]
    fn boundary_on_a_tool_result_does_not_move_to_a_following_user_message() {
        let translated = translated(json!({
            "model": "claude-opus-5",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "call-2", "type": "function", "function": {"name": "b", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call-1", "content": "first result"},
                {"role": "tool", "tool_call_id": "call-2", "content": "second result",
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"role": "user", "content": "volatile per-turn text"}
            ]
        }));

        assert_eq!(
            translated["messages"][2]["content"],
            json!([
                {"type": "tool_result", "tool_use_id": "call-1", "content": "first result"},
                {"type": "tool_result", "tool_use_id": "call-2", "content": "second result",
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"type": "text", "text": "volatile per-turn text"}
            ])
        );
    }

    #[test]
    fn boundary_on_an_assistant_text_stays_put_when_another_assistant_text_follows() {
        let translated = translated(json!({
            "model": "claude-opus-5",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "stable answer",
                 "cache_control": {"type": "ephemeral"}},
                {"role": "assistant", "content": "later addition"},
                {"role": "assistant", "content": ""}
            ]
        }));

        assert_eq!(
            translated["messages"][1]["content"],
            json!([
                {"type": "text", "text": "stable answer", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "later addition"}
            ])
        );
    }

    #[test]
    fn four_boundaries_fit_in_merged_messages_and_a_fifth_is_rejected() {
        let conversation = |last: serde_json::Value| {
            json!({
                "model": "claude-opus-5",
                "messages": [
                    {"role": "user", "content": "u1", "cache_control": {"type": "ephemeral"}},
                    {"role": "user", "content": "u2", "cache_control": {"type": "ephemeral"}},
                    {"role": "assistant", "content": "a1", "cache_control": {"type": "ephemeral"}},
                    {"role": "assistant", "content": "a2", "cache_control": {"type": "ephemeral"}},
                    last
                ]
            })
        };
        let marker = json!({"type": "ephemeral"});

        let four = translated(conversation(json!({"role": "user", "content": "u3"})));
        assert_eq!(
            markers(&four),
            vec![
                vec![marker.clone(), marker.clone()],
                vec![marker.clone(), marker.clone()],
                vec![json!(null)],
            ]
        );

        let error = translation_error(conversation(
            json!({"role": "user", "content": "u3", "cache_control": {"type": "ephemeral"}}),
        ));
        assert!(error.contains("at most 4 cache breakpoints"), "{error}");
        assert!(error.contains("produces 5"), "{error}");
    }

    #[test]
    fn one_hour_boundary_after_a_five_minute_boundary_is_rejected() {
        let short = json!({"type": "ephemeral"});
        let explicit_short = json!({"type": "ephemeral", "ttl": "5m"});
        let long = json!({"type": "ephemeral", "ttl": "1h"});

        // Across messages.
        let error = translation_error(json!({"model": "m", "messages": [
            {"role": "user", "content": "u1", "cache_control": short},
            {"role": "assistant", "content": "a1", "cache_control": long}
        ]}));
        assert!(error.contains("messages[1].cache_control.ttl"), "{error}");
        assert!(error.contains("messages[0]"), "{error}");
        assert!(error.contains("1-hour boundaries before 5-minute ones"), "{error}");

        // Inside one merged message, and with an explicit \"5m\".
        let error = translation_error(json!({"model": "m", "messages": [
            {"role": "user", "content": "u1", "cache_control": explicit_short},
            {"role": "user", "content": "u2", "cache_control": long}
        ]}));
        assert!(error.contains("messages[1].cache_control.ttl"), "{error}");

        // System text is first in the prompt wherever it sits in the list.
        let error = translation_error(json!({"model": "m", "messages": [
            {"role": "system", "content": "s", "cache_control": short},
            {"role": "user", "content": "u1", "cache_control": long}
        ]}));
        assert!(error.contains("messages[1].cache_control.ttl"), "{error}");
        let error = translation_error(json!({"model": "m", "messages": [
            {"role": "user", "content": "u1", "cache_control": long},
            {"role": "system", "content": "s", "cache_control": short}
        ]}));
        assert!(error.contains("messages[0].cache_control.ttl"), "{error}");
        assert!(error.contains("messages[1]"), "{error}");
    }

    #[test]
    fn one_hour_boundaries_before_five_minute_boundaries_are_accepted() {
        let translated = translated(json!({"model": "m", "messages": [
            {"role": "user", "content": "u1", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"role": "system", "content": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"role": "user", "content": "u2", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"role": "assistant", "content": "a1", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
            {"role": "user", "content": "u3"}
        ]}));

        assert_eq!(
            translated["system"][1]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert_eq!(
            markers(&translated),
            vec![
                vec![json!({"type": "ephemeral", "ttl": "1h"}), json!({"type": "ephemeral", "ttl": "1h"})],
                vec![json!({"type": "ephemeral", "ttl": "5m"})],
                vec![json!(null)],
            ]
        );
    }

    #[test]
    fn boundary_on_a_contentless_message_shares_the_block_merged_before_it() {
        let conversation = |first: serde_json::Value, second: serde_json::Value| {
            json!({"model": "m", "messages": [
                {"role": "user", "content": "one", "cache_control": first},
                {"role": "user", "content": null, "cache_control": second},
                {"role": "assistant", "content": "a1", "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": "u2", "cache_control": {"type": "ephemeral"}},
                {"role": "assistant", "content": "a2", "cache_control": {"type": "ephemeral"}}
            ]})
        };
        let short = json!({"type": "ephemeral"});

        // Two source boundaries, one block, one breakpoint: four in total.
        let shared = translated(conversation(short.clone(), short.clone()));
        assert_eq!(
            markers(&shared),
            vec![vec![short.clone()], vec![short.clone()], vec![short.clone()], vec![short.clone()]]
        );

        let error = translation_error(conversation(
            json!({"type": "ephemeral", "ttl": "1h"}),
            short.clone(),
        ));
        assert!(error.contains("messages[1].cache_control cannot be represented"), "{error}");
        assert!(error.contains("different ttl"), "{error}");

        // With nothing merged before it there is no block to carry the boundary.
        let error = translation_error(json!({"model": "m", "messages": [
            {"role": "user", "content": null, "cache_control": short}
        ]}));
        assert!(error.contains("message has no content blocks"), "{error}");
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
    fn boundaries_on_two_merged_user_messages_become_two_block_markers() {
        let translated = translated(json!({
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "one", "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": "two", "cache_control": {"type": "ephemeral"}}
            ]
        }));

        assert_eq!(
            translated["messages"],
            json!([{"role": "user", "content": [
                {"type": "text", "text": "one", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "two", "cache_control": {"type": "ephemeral"}}
            ]}])
        );
    }

    /// Requests whose exact upstream bytes must not change: what Adapsis sends
    /// today (boundaries on system text and assistant messages), boundaries on
    /// the last message of a merged run, and the legacy heuristic. `expected`
    /// was recorded from the translation at c4f0d36, before boundaries were
    /// placed per source message.
    #[test]
    fn pinned_requests_translate_byte_identically() {
        let pins: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/anthropic_request_pins.json")).unwrap();
        let pins = pins.as_array().unwrap();
        assert_eq!(pins.len(), 13);
        for pin in pins {
            let name = pin["name"].as_str().unwrap();
            let request: OpenAIRequest = serde_json::from_value(pin["request"].clone()).unwrap();
            let translated = translate_request(&request).unwrap_or_else(|e| panic!("{name}: {e}"));
            let actual = serde_json::to_string(&translated).unwrap();
            assert_eq!(actual, pin["expected"].as_str().unwrap(), "pin {name} changed");
        }
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
