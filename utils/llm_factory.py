import os


def use_bedrock() -> bool:
    return os.getenv("USE_BEDROCK", "false").lower() == "true"


def get_model_id() -> str:
    if use_bedrock():
        return os.getenv("BEDROCK_PRIMARY_MODEL", "us.anthropic.claude-sonnet-4-5-20250514-v1:0")
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


def get_anthropic_client():
    """Returns AnthropicBedrock when USE_BEDROCK=true, Anthropic otherwise."""
    import anthropic
    if use_bedrock():
        return anthropic.AnthropicBedrock(aws_region=os.getenv("AWS_REGION", "us-east-1"))
    return anthropic.Anthropic()


def get_langchain_model(max_tokens: int = 1024):
    """Returns ChatBedrockConverse when USE_BEDROCK=true, ChatAnthropic otherwise."""
    if use_bedrock():
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(
            model=get_model_id(),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            max_tokens=max_tokens,
        )
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=get_model_id(), max_tokens=max_tokens)
