import pytest
import os
import sys
import pathlib
from unittest.mock import patch, MagicMock
import asyncio
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from workflow.tools import it_seems_official, norm, collection_exists, InvalidFrameworkNameException, AmbiguousFrameworkNameException, is_valid_doc_url

### it_seems_official
# (True, URL) 형식의 테스트 케이스
OFFICIAL_URLS = [
    ("https://spring.io/guides/gs/spring-boot/"),
    ("https://react.dev/learn/start-a-new-react-project"),
    ("https://pandas.pydata.org/docs/user_guide/index.html")
]

# (False, URL) 형식의 테스트 케이스
NON_OFFICIAL_URLS = [
    ("https://github.com/spring-projects/spring-boot"),
    ("https://medium.com/@someone/how-to-use-react-18-123456"),
    ("https://stackoverflow.com/questions/12345678/how-to-fix-error"),
    ("https://mysite.dev.to/tutorial/install-nodejs"),
    ("https://en.wikipedia.org/wiki/React_(software)"),
    ("https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
]

@pytest.mark.parametrize("url", OFFICIAL_URLS)
def test_it_seems_official_true(url):
    """공식 문서로 보이는 URL은 True를 리턴해야 한다."""
    assert it_seems_official(url) is True

@pytest.mark.parametrize("url", NON_OFFICIAL_URLS)
def test_it_seems_official_false(url):
    """공식 문서가 아니거나 제외 대상 키워드를 포함하면 False를 리턴해야 한다."""
    assert it_seems_official(url) is False

### norm
@pytest.mark.parametrize("word,answer", [
    ("Spring Boot", "springboot"),
    ("Spring-Boot", "springboot"),
    ("spring.boot", "springboot"),
    ("spring  boot", "springboot"),
    (" spring Boo t", "springboot")
])
def test_norm(word, answer):
    assert norm(word) == answer

def test_norm_invalid():
    with pytest.raises(InvalidFrameworkNameException):
        norm("%^@$. ")

### collection_exists
@patch('workflow.tools.client')
def test_collection_exists_exact_match(mock_client):
    mock_client.list_collections.return_value = ["Example_Collection"]
    assert collection_exists("example collection") is True

@patch('workflow.tools.client')
def test_collection_exists_no_match(mock_client):
    mock_client.list_collections.return_value = ["unrelated"]
    assert collection_exists("somethingelse") is False

@patch('workflow.tools.client')
def test_collection_exists_special_characters(mock_client):
    mock_client.list_collections.return_value = ["data-set_2024"]
    assert collection_exists("Data Set 2024") is True

@patch('workflow.tools.client')
@pytest.mark.parametrize("return_value,collection_name", [
    (["spring framework", "spring boot", "spring mvc"], "spring"),
    (["spring"], "spring framework"),
    (["react"], "react.js"),
    (["reactjs"], "react"),
    (["test"], "es")
])
def test_collection_exists_ambiguous_keywords(mock_client, return_value, collection_name):
    mock_client.list_collections.return_value = return_value
    with pytest.raises(AmbiguousFrameworkNameException):
        collection_exists(collection_name)

@patch('workflow.tools.client')
@pytest.mark.parametrize("return_value,collection_name", [
    (["spring"], "spring boot"),     # 허용됨
    (["react"], "react native"),     # 허용됨
    (["vue"], "vue router"),         # 허용됨
    (["flask"], "flask login"),      # 허용됨
])
def test_collection_exists_excluded_ambiguous_pair_passes(mock_client, return_value, collection_name):
    mock_client.list_collections.return_value = return_value
    assert collection_exists(collection_name) is False

@patch('workflow.tools.client')
def test_collection_exists_with_normalized_name(mock_client):
    mock_client.list_collections.return_value = ["ReactNative"]
    assert collection_exists("react native") is True

@patch('workflow.tools.client')
def test_collection_exists_ambiguous_due_to_normalized_match(mock_client):
    mock_client.list_collections.return_value = ["vuex"]
    with pytest.raises(AmbiguousFrameworkNameException):
        collection_exists("vue")

@patch('workflow.tools.client')
def test_collection_exists_conflict_with_short_existing_name(mock_client):
    mock_client.list_collections.return_value = ["spring"]
    with pytest.raises(AmbiguousFrameworkNameException) as e:
        collection_exists("spring framework")
    assert "짧은 이름" in str(e.value)

## is_valid_doc_url
@pytest.mark.parametrize("href, expected", [
    # 유효한 문서형 링크
    ("/guide/", True),
    ("/guide/introduction.html", True),
    ("/guide/introduction", True),  # 확장자 없음
    ("getting-started", True),      # 상대경로, 확장자 없음
    ("https://example.com/docs", True),
    # 제외 대상
    ("#section", False),  # 앵커
    ("javascript:void(0)", False),  # JS
    ("/assets/logo.png", False),    # 이미지
    ("/scripts/app.js", False),     # JS 파일
    ("/downloads/manual.pdf", False),  # PDF
    ("", False),  # 빈 문자열
    # 의도적으로 긴 경로도 포함
    ("/docs/some/page/without/extension", True),
    # 확장자 있는 정적 리소스지만 '/' 포함된 경우
    ("/docs/script.min.js/", False),
])
def test_is_valid_doc_url(href, expected):
    assert is_valid_doc_url(href) == expected