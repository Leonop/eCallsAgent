
# OpenAI API compatibility module to support both v0.x and v1.x+ versions
import importlib.util
import os
import warnings

# Check which version of OpenAI is installed
def get_openai_version():
    try:
        import openai
        version = getattr(openai, '__version__', '0.0.0')
        return version
    except ImportError:
        return None

def is_v1_or_higher():
    version = get_openai_version()
    if not version:
        return False
    return version.startswith(('1.', '2.'))

# Create a compatible client
def create_openai_client(api_key=None):
    import openai
    
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if is_v1_or_higher():
        # Modern OpenAI client (v1.x+)
        return openai.OpenAI(api_key=api_key)
    else:
        # Legacy OpenAI client (v0.x)
        openai.api_key = api_key
        return openai

# Compatible completion function that works with both API versions
def create_completion(client, model="gpt-3.5-turbo", messages=None, prompt=None, **kwargs):
    if is_v1_or_higher():
        # Modern API expects messages
        if messages is None and prompt is not None:
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    else:
        # Legacy API with different parameters
        if prompt is None and messages is not None:
            # Extract prompt from messages for legacy API
            prompt = " ".join([m["content"] for m in messages])
        
        response = client.ChatCompletion.create(
            model=model,
            messages=messages if messages else [{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message["content"]
