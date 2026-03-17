use fermi_io::ModelArch;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptFormat {
    ChatMl,
    Llama3,
}

#[derive(Debug, Clone, Copy)]
pub struct PromptMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

pub fn build_stop_tokens(arch: ModelArch, tokenizer: &Tokenizer) -> Vec<u32> {
    let token_names: &[&str] = match prompt_format_for_arch(arch) {
        PromptFormat::ChatMl => &["<|im_end|>", "<|endoftext|>"],
        PromptFormat::Llama3 => &["<|eot_id|>", "<|end_of_text|>"],
    };
    token_names
        .iter()
        .filter_map(|name| tokenizer.token_to_id(name))
        .fold(Vec::new(), |mut acc, token_id| {
            if !acc.contains(&token_id) {
                acc.push(token_id);
            }
            acc
        })
}

pub fn assistant_turn_end_token_id(arch: ModelArch, tokenizer: &Tokenizer) -> Option<u32> {
    let token_name = match prompt_format_for_arch(arch) {
        PromptFormat::ChatMl => "<|im_end|>",
        PromptFormat::Llama3 => "<|eot_id|>",
    };
    tokenizer.token_to_id(token_name)
}

pub fn render_chat_prompt(arch: ModelArch, user_text: &str, system_prompt: Option<&str>) -> String {
    render_history_prompt(arch, &[], user_text, system_prompt)
}

pub fn render_history_prompt(
    arch: ModelArch,
    pairs: &[(String, String)],
    user_text: &str,
    system_prompt: Option<&str>,
) -> String {
    let mut out = String::new();
    let format = prompt_format_for_arch(arch);
    begin_prompt(&mut out, format);
    if let Some(sys) = system_prompt.map(str::trim).filter(|sys| !sys.is_empty()) {
        push_message(&mut out, format, "system", sys);
    }
    for (user, assistant) in pairs {
        push_message(&mut out, format, "user", user);
        push_message(&mut out, format, "assistant", assistant);
    }
    push_message(&mut out, format, "user", user_text);
    push_assistant_prefix(&mut out, format);
    out
}

pub fn render_user_chunk(
    arch: ModelArch,
    user_text: &str,
    has_context: bool,
    system_prompt: Option<&str>,
) -> String {
    let mut out = String::new();
    let format = prompt_format_for_arch(arch);
    if !has_context {
        begin_prompt(&mut out, format);
        if let Some(sys) = system_prompt.map(str::trim).filter(|sys| !sys.is_empty()) {
            push_message(&mut out, format, "system", sys);
        }
    } else if matches!(format, PromptFormat::ChatMl) {
        out.push('\n');
    }
    push_message(&mut out, format, "user", user_text);
    push_assistant_prefix(&mut out, format);
    out
}

pub fn render_messages_prompt(
    arch: ModelArch,
    messages: &[PromptMessage<'_>],
    default_system_prompt: Option<&str>,
    extra_system_prompt: Option<&str>,
) -> String {
    let mut out = String::new();
    let format = prompt_format_for_arch(arch);
    begin_prompt(&mut out, format);

    let has_explicit_system = messages
        .iter()
        .any(|msg| matches!(normalize_role(msg.role), Some("system")));
    if !has_explicit_system {
        if let Some(sys) = default_system_prompt.map(str::trim).filter(|sys| !sys.is_empty()) {
            push_message(&mut out, format, "system", sys);
        }
    }

    for msg in messages {
        let Some(role) = normalize_role(msg.role) else {
            continue;
        };
        let content = msg.content.trim();
        if content.is_empty() {
            continue;
        }
        push_message(&mut out, format, role, content);
    }

    if let Some(sys) = extra_system_prompt.map(str::trim).filter(|sys| !sys.is_empty()) {
        push_message(&mut out, format, "system", sys);
    }

    push_assistant_prefix(&mut out, format);
    out
}

fn prompt_format_for_arch(arch: ModelArch) -> PromptFormat {
    match arch {
        ModelArch::Llama => PromptFormat::Llama3,
        ModelArch::Qwen | ModelArch::Phi3 => PromptFormat::ChatMl,
    }
}

fn normalize_role(role: &str) -> Option<&'static str> {
    match role {
        "system" | "developer" => Some("system"),
        "user" => Some("user"),
        "assistant" => Some("assistant"),
        _ => None,
    }
}

fn begin_prompt(out: &mut String, format: PromptFormat) {
    if matches!(format, PromptFormat::Llama3) {
        out.push_str("<|begin_of_text|>");
    }
}

fn push_message(out: &mut String, format: PromptFormat, role: &str, content: &str) {
    match format {
        PromptFormat::ChatMl => {
            out.push_str("<|im_start|>");
            out.push_str(role);
            out.push('\n');
            out.push_str(content);
            out.push_str("<|im_end|>\n");
        }
        PromptFormat::Llama3 => {
            out.push_str("<|start_header_id|>");
            out.push_str(role);
            out.push_str("<|end_header_id|>\n\n");
            out.push_str(content);
            out.push_str("<|eot_id|>");
        }
    }
}

fn push_assistant_prefix(out: &mut String, format: PromptFormat) {
    match format {
        PromptFormat::ChatMl => out.push_str("<|im_start|>assistant\n"),
        PromptFormat::Llama3 => {
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_chat_prompt_uses_chatml() {
        let prompt = render_chat_prompt(ModelArch::Qwen, "hello", Some("sys"));
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama_chat_prompt_uses_llama3_template() {
        let prompt = render_chat_prompt(ModelArch::Llama, "hello", Some("sys"));
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn llama_user_chunk_avoids_chatml_newline_prefix() {
        let chunk = render_user_chunk(ModelArch::Llama, "hello", true, None);
        assert!(!chunk.starts_with('\n'));
        assert!(chunk.starts_with("<|start_header_id|>user<|end_header_id|>"));
    }
}
