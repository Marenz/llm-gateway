use crate::types::{
    AnthropicContentBlock, AnthropicMessage, AnthropicMessageContent, AnthropicNamedToolChoice,
    AnthropicRequest, AnthropicSystem, AnthropicSystemMessage, AnthropicTool, AnthropicToolChoice,
    AnthropicToolChoiceMode, OpenAIMessage, OpenAIMessageContent, OpenAIStop, OpenAIToolChoice,
};

/// Required system prompt prefix for Anthropic OAuth tokens to unlock non-haiku models.
const OAUTH_REQUIRED_SYSTEM: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

pub fn translate_request(req: &crate::types::OpenAIRequest) -> crate::types::AnthropicRequest {
    let system = collect_system_prompt(&req.messages);
    let mut messages = Vec::new();

    for message in &req.messages {
        match message.role {
            crate::types::OpenAIRole::System => {}
            crate::types::OpenAIRole::User => push_message(
                &mut messages,
                AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::Text(extract_text(&message.content)),
                },
            ),
            crate::types::OpenAIRole::Assistant => {
                push_message(&mut messages, translate_assistant_message(message));
            }
            crate::types::OpenAIRole::Tool => {
                if let Some(tool_call_id) = &message.tool_call_id {
                    push_message(
                        &mut messages,
                        AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicMessageContent::Blocks(vec![
                                AnthropicContentBlock::ToolResult {
                                    tool_use_id: tool_call_id.clone(),
                                    content: serde_json::Value::String(extract_text(
                                        &message.content,
                                    )),
                                },
                            ]),
                        },
                    );
                }
            }
        }
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

    AnthropicRequest {
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
    }
}

fn collect_system_prompt(messages: &[OpenAIMessage]) -> Option<AnthropicSystem> {
    // Always prepend the required OAuth system prompt so non-haiku models work.
    let mut blocks = vec![AnthropicSystemMessage {
        kind: "text".to_string(),
        text: OAUTH_REQUIRED_SYSTEM.to_string(),
        cache_control: None,
    }];

    for message in messages
        .iter()
        .filter(|m| matches!(m.role, crate::types::OpenAIRole::System))
    {
        let text = extract_text(&message.content);
        if !text.is_empty() {
            blocks.push(AnthropicSystemMessage {
                kind: "text".to_string(),
                text,
                cache_control: None,
            });
        }
    }

    Some(AnthropicSystem::Messages(blocks))
}

fn translate_assistant_message(message: &OpenAIMessage) -> AnthropicMessage {
    let mut blocks = Vec::new();
    let text = extract_text(&message.content);

    if !text.is_empty() {
        blocks.push(AnthropicContentBlock::Text { text });
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
            });
        }
    }

    AnthropicMessage {
        role: "assistant".to_string(),
        content: match blocks.as_slice() {
            [] => AnthropicMessageContent::Text(String::new()),
            [AnthropicContentBlock::Text { text }] => AnthropicMessageContent::Text(text.clone()),
            _ => AnthropicMessageContent::Blocks(blocks),
        },
    }
}

fn extract_text(content: &OpenAIMessageContent) -> String {
    match content {
        OpenAIMessageContent::Text(text) => text.clone(),
        OpenAIMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                crate::types::OpenAIContentPart::Text { text } => Some(text.as_str()),
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

fn push_message(messages: &mut Vec<AnthropicMessage>, message: AnthropicMessage) {
    if let Some(last) = messages.last_mut() {
        if same_role(&last.role, &message.role) {
            last.content = merge_content(
                std::mem::replace(
                    &mut last.content,
                    AnthropicMessageContent::Text(String::new()),
                ),
                message.content,
            );
            return;
        }
    }

    messages.push(message);
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
                vec![AnthropicContentBlock::Text { text }]
            }
        }
        AnthropicMessageContent::Blocks(blocks) => blocks,
    }
}
