use std::collections::HashMap;

use crate::config::{GatewayConfig, ProviderConfig};
use crate::types::{ProviderKind, ResolvedModel};

pub struct ModelResolver {
    pub routes: Vec<ModelRoute>,
    pub aliases: HashMap<String, String>,
}

pub struct ModelRoute {
    pub pattern: String,
    pub provider_kind: ProviderKind,
    pub provider_name: String,
    pub strip_prefix: Option<String>,
}

impl ModelResolver {
    pub fn from_config(config: &GatewayConfig) -> Self {
        let mut routes = Vec::new();
        let mut implicit = ImplicitProviders::default();

        for provider in &config.providers {
            match provider {
                ProviderConfig::Anthropic(cfg) => {
                    implicit.anthropic.get_or_insert_with(|| cfg.name.clone());
                    for model in &cfg.models {
                        routes.push(ModelRoute {
                            pattern: model.clone(),
                            provider_kind: ProviderKind::Anthropic,
                            provider_name: cfg.name.clone(),
                            strip_prefix: None,
                        });
                    }
                }
                ProviderConfig::Chatgpt(cfg) => {
                    implicit.chatgpt.get_or_insert_with(|| cfg.name.clone());
                    for model in &cfg.models {
                        routes.push(ModelRoute {
                            pattern: model.clone(),
                            provider_kind: ProviderKind::Chatgpt,
                            provider_name: cfg.name.clone(),
                            strip_prefix: None,
                        });
                    }
                }
                ProviderConfig::Openai(cfg) => {
                    implicit.openai.get_or_insert_with(|| cfg.name.clone());
                    for model in &cfg.models {
                        routes.push(ModelRoute {
                            pattern: model.clone(),
                            provider_kind: ProviderKind::Openai,
                            provider_name: cfg.name.clone(),
                            strip_prefix: None,
                        });
                    }
                }
                ProviderConfig::XiaomiMimo(cfg) => {
                    implicit.mimo.get_or_insert_with(|| cfg.name.clone());
                    for model in &cfg.models {
                        routes.push(ModelRoute {
                            pattern: model.clone(),
                            provider_kind: ProviderKind::XiaomiMimo,
                            provider_name: cfg.name.clone(),
                            strip_prefix: None,
                        });
                    }
                }
                ProviderConfig::OpenaiCompatible(cfg) => {
                    for model in &cfg.models {
                        routes.push(ModelRoute {
                            pattern: model.clone(),
                            provider_kind: ProviderKind::OpenaiCompatible,
                            provider_name: cfg.name.clone(),
                            strip_prefix: None,
                        });
                    }
                }
            }
        }

        if let Some(provider_name) = implicit.anthropic {
            routes.push(ModelRoute {
                pattern: "anthropic/".to_string(),
                provider_kind: ProviderKind::Anthropic,
                provider_name,
                strip_prefix: Some("anthropic/".to_string()),
            });
        }

        if let Some(provider_name) = implicit.chatgpt {
            routes.push(ModelRoute {
                pattern: "chatgpt/".to_string(),
                provider_kind: ProviderKind::Chatgpt,
                provider_name,
                strip_prefix: Some("chatgpt/".to_string()),
            });
        }

        if let Some(provider_name) = implicit.openai {
            routes.push(ModelRoute {
                pattern: "openai/".to_string(),
                provider_kind: ProviderKind::Openai,
                provider_name,
                strip_prefix: Some("openai/".to_string()),
            });
        }

        if let Some(provider_name) = implicit.mimo {
            routes.push(ModelRoute {
                pattern: "mimo/".to_string(),
                provider_kind: ProviderKind::XiaomiMimo,
                provider_name: provider_name.clone(),
                strip_prefix: Some("mimo/".to_string()),
            });
            routes.push(ModelRoute {
                pattern: "xiaomi_mimo/".to_string(),
                provider_kind: ProviderKind::XiaomiMimo,
                provider_name,
                strip_prefix: Some("xiaomi_mimo/".to_string()),
            });
        }

        Self {
            routes,
            aliases: config.model_aliases.clone(),
        }
    }

    pub fn resolve(&self, model: &str) -> Option<ResolvedModel> {
        let resolved_name = self.aliases.get(model).map(String::as_str).unwrap_or(model);

        self.routes.iter().find_map(|route| {
            let upstream_model = if let Some(prefix) = &route.strip_prefix {
                resolved_name.strip_prefix(prefix).map(str::to_string)
            } else if resolved_name == route.pattern {
                Some(resolved_name.to_string())
            } else {
                None
            }?;

            Some(ResolvedModel {
                provider_kind: route.provider_kind.clone(),
                provider_name: route.provider_name.clone(),
                upstream_model,
            })
        })
    }
}

#[derive(Default)]
struct ImplicitProviders {
    anthropic: Option<String>,
    chatgpt: Option<String>,
    openai: Option<String>,
    mimo: Option<String>,
}
