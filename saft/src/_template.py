"""
!!! Label is in "format_assistant"
"""
from llamafactory.data.template import (
    register_template,
    StringFormatter,
    EmptyFormatter,
    ToolFormatter,
    FunctionFormatter,
    Llama2Template,
    get_mm_plugin
)

def register_new_template():
    register_template(
        name="llama2-it-sys",
        format_user=StringFormatter(slots=[{"bos_token"}, "[INST] {{content}} [/INST]"]),
        format_system=StringFormatter(slots=["<<SYS>>\n{{content}}\n<</SYS>>\n\n"]),
        default_system="You are a helpful assistant.",
        template_class=Llama2Template,
    )
    register_template(
        name="llama2-pt-sys",
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_user=StringFormatter(slots=["{{content}}\n\n"]),
        format_assistant=StringFormatter(slots=["{{content}}"]),
        default_system="You are a helpful assistant.",
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        template_class=Llama2Template,
    )

    # from template deepseek
    register_template(
        name="deepseekmath-it-sys",
        format_user=StringFormatter(slots=["User: {{content}}\n\nAssistant:"]),
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        default_system="You are a helpful assistant.",
    )

    register_template(
        name="deepseekmath-pt-sys",
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_user=StringFormatter(slots=["{{content}}\n\n"]),
        format_assistant=StringFormatter(slots=["{{content}}"]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        default_system="You are a helpful assistant.",
    )


    register_template(
        name="llama3-it-sys-cot",
        format_user=StringFormatter(
            slots=[
                (
                    "<|start_header_id|>user<|end_header_id|>\n\n{{content}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            ]
        ),
        format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
        format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
        format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
        format_observation=StringFormatter(
            slots=[
                (
                    "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            ]
        ),
        default_system="You are a helpful assistant.",
        format_tools=ToolFormatter(tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        replace_eos=True,
    )

    register_template(
        name="llama3-pt-sys",
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_user=StringFormatter(slots=["{{content}}\n\n"]),
        format_assistant=StringFormatter(slots=["{{content}}"]),
        format_observation=EmptyFormatter(slots=[{"bos_token"}]),
        default_system="You are a helpful assistant.",
        format_tools=ToolFormatter(tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        replace_eos=True,
    )

    register_template(
        name="llama3-it-sys",
        format_user=StringFormatter(
            slots=[
                (
                    "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            ]
        ),
        format_assistant=StringFormatter(slots=["{{content}}<|eot_id|>"]),
        format_system=StringFormatter(slots=["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]),
        format_function=FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3"),
        format_observation=StringFormatter(
            slots=[
                (
                    "<|start_header_id|>ipython<|end_header_id|>\n\n{{content}}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            ]
        ),
        default_system="You are a helpful assistant.",
        format_tools=ToolFormatter(tool_format="llama3"),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        replace_eos=True,
    )

    register_template(
        name="qwen3-it-sys",
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
        ),
        default_system="You are a helpful assistant.",
        format_tools=ToolFormatter(tool_format="qwen"),
        stop_words=["<|im_end|>"],
        replace_eos=True,
    )

    register_template(
        name="qwen3-pt-sys",
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_user=StringFormatter(slots=["{{content}}\n\n"]),
        format_assistant=StringFormatter(slots=["{{content}}"]),
        format_observation=EmptyFormatter(slots=[{"bos_token"}]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        format_tools=ToolFormatter(tool_format="qwen"),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        replace_eos=True,
    )

    register_template(
        name="gemma3-it-besa",
        format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_observation=StringFormatter(
            slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
        ),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<end_of_turn>"],
        replace_eos=True,
        mm_plugin=get_mm_plugin("gemma3", image_token="<image_soft_token>"),
        template_class=Llama2Template,
    )

    register_template(
        name="gemma3-it-sys",
        format_user=StringFormatter(slots=["<start_of_turn>user\nYou are a helpful assistant.\n\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
        format_system=EmptyFormatter(slots=[""]),
        format_observation=StringFormatter(
            slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
        ),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<eos>", "<end_of_turn>"],
        efficient_eos=True,
        template_class=Llama2Template,
    )

    register_template(
        name="gemma3-pt-sys",
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_user=StringFormatter(slots=["{{content}}\n\n"]),
        format_assistant=StringFormatter(slots=["{{content}}"]),
        format_observation=EmptyFormatter(slots=[{"bos_token"}]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<eos>", "<end_of_turn>"],
        efficient_eos=True,
        template_class=Llama2Template,
    )

    # sep：Gemma-2
    # Gemma-2
    register_template(
        name="gemma2-default",
        format_user=StringFormatter(slots=["{{content}}"]),
        format_assistant=StringFormatter(slots=["{{content}}"]),
        format_system=StringFormatter(slots=["{{content}}"]),
        format_observation=EmptyFormatter(slots=[{"bos_token"}]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<eos>"],
        efficient_eos=True,
        template_class=Llama2Template,
    )

    # Gemma-2 Chat-Template
    register_template(
        name="gemma2-it",
        format_user=StringFormatter(slots=["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_observation=StringFormatter(
            slots=["<start_of_turn>tool\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
        ),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<eos>", "<end_of_turn>"],
        efficient_eos=True,
        template_class=Llama2Template,
    )

    # gemma2 w. metamath prompt w.o. chat template
    register_template(
        name="gemma2-metamath",
        format_user=StringFormatter(
            slots=["Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{{content}}\n\n### Response: Let's think step by step. "]
        ),
        format_assistant=StringFormatter(slots=["{{content}}"]),
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_observation=EmptyFormatter(slots=[{"bos_token"}]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<eos>", "<end_of_turn>"],
        efficient_eos=True,
        template_class=Llama2Template,
    )

    # gemma2 w. metamath prompt w. chat template
    register_template(
        name="gemma2-it-metamath",
        format_user=StringFormatter(
            slots=["<start_of_turn>user\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{{content}}<end_of_turn>\n\n<start_of_turn>model\nLet's think step by step. "]
        ),
        format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\n"]),
        format_system=StringFormatter(slots=["{{content}}\n\n"]),
        format_observation=EmptyFormatter(slots=[{"bos_token"}]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<eos>", "<end_of_turn>"],
        efficient_eos=True,
        template_class=Llama2Template,
    )