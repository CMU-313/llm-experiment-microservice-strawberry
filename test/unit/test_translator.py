from src.translator import translate_content


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

def test_llm_normal_response():
    text = "This is already in English."

    is_english, translated_content = translate_content(text)

    assert is_english is True
    assert translated_content == text
    text = "Hola, soy Dora!"

    is_english, translated_content = translate_content(text)

    assert is_english is False
    # assert translated_content == text

def test_llm_gibberish_response():
    text = "asdkj lqweoi!! 123 ??"

    is_english, translated_content = translate_content(text)

    assert translated_content == text

    assert is_english in (True, False)