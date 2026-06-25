from pathlib import Path
import tomllib


def test_news_impact_full_package_importable_with_internal_imports():
    from src.news_impact.pipeline import DailyPipelineInputs, run_daily_pipeline
    from src.news_impact.schema import ImpactEvent, NewsItem
    from src.news_impact.stock_factors.classifier import classify_factors

    assert DailyPipelineInputs.__name__ == "DailyPipelineInputs"
    assert callable(run_daily_pipeline)
    assert ImpactEvent.__name__ == "ImpactEvent"
    assert NewsItem.__name__ == "NewsItem"
    assert callable(classify_factors)


def test_pyproject_package_discovery_includes_migrated_news_impact_package():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)

    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert package_find["where"] == ["."]
    assert package_find["include"] == ["src*"]
    assert pyproject["project"]["scripts"]["stock-news-impact"] == "src.news_impact.run:main"


def test_news_impact_runtime_examples_are_migrated_without_private_config():
    assert Path("configs/news_impact.example.json").exists()
    assert Path("configs/news_impact.gemma.example.json").exists()
    assert Path("configs/news_impact.openai.example.json").exists()
    assert not Path("configs/news_impact.json").exists()
    assert Path("data/news_impact/watchlist.example.csv").exists()
    assert Path("data/news_impact/company_master.example.csv").exists()


def test_news_impact_runtime_examples_default_to_gemma_with_openai_option():
    default_example = Path("configs/news_impact.example.json").read_text(encoding="utf-8")
    gemma_example = Path("configs/news_impact.gemma.example.json").read_text(encoding="utf-8")
    openai_example = Path("configs/news_impact.openai.example.json").read_text(encoding="utf-8")

    # 기본 템플릿(example.json)은 코드 기본값과 동일한 로컬 gemma를 가리킨다.
    assert '"llm_provider": "llama_cpp"' in default_example
    assert '"llm_model": "gemma-4-26b-a4b"' in default_example
    assert '"llm_provider": "llama_cpp"' in gemma_example
    assert '"llm_model": "gemma-4-26b-a4b"' in gemma_example
    # OpenAI는 선택지로 보존하며 키는 OPENAI_API_KEY 환경변수에서 읽는다.
    assert '"llm_provider": "openai"' in openai_example
    assert '"llm_model": "gpt-5-mini"' in openai_example
    assert "OPENAI_API_KEY" not in openai_example


def test_news_impact_llm_prompt_asset_is_vendored():
    from pathlib import Path

    # impact_judge.build_system_prompt()가 런타임에 읽는 필수 자산. 머지 시 누락 방지.
    assert Path("docs/NEWS_IMPACT_LLM_PROMPT.md").exists()



def test_daily_pipeline_inputs_accepts_stable_llm_cache_dir(tmp_path):
    from src.news_impact.pipeline import DailyPipelineInputs

    inputs = DailyPipelineInputs(
        run_date="2026-06-25",
        watchlist_path=tmp_path / "wl.csv",
        company_master_path=tmp_path / "cm.csv",
        input_fixture_path=tmp_path / "fx.json",
        output_dir=tmp_path / "out",
        llm_cache_dir=tmp_path / "stable_cache",
    )
    assert Path(inputs.llm_cache_dir) == tmp_path / "stable_cache"



def _news_stub(title, summary="", raw_text="", ticker=""):
    from types import SimpleNamespace

    return SimpleNamespace(title=title, summary=summary, raw_text=raw_text, ticker=ticker)


def test_target_tickers_narrows_to_company_name_match():
    from src.news_impact.pipeline import _target_tickers_for_news

    companies = {
        "005930": {"company": "????"},
        "000660": {"company": "SK????"},
    }
    item = _news_stub("????, ?? HBM ??")
    assert _target_tickers_for_news(item, ["005930", "000660"], companies) == ["005930"]


def test_target_tickers_falls_back_to_full_watchlist_when_no_match():
    from src.news_impact.pipeline import _target_tickers_for_news

    companies = {"005930": {"company": "????"}, "000660": {"company": "SK????"}}
    item = _news_stub("??? ??? ??? ??")
    assert _target_tickers_for_news(item, ["005930", "000660"], companies) == ["005930", "000660"]



def test_article_text_is_truncated_with_flag():
    from types import SimpleNamespace
    from src.news_impact.pipeline import _llm_article_text_and_flags, MAX_ARTICLE_CHARS

    long_item = SimpleNamespace(
        quality_flags=(), raw_text="?" * (MAX_ARTICLE_CHARS + 500), summary=""
    )
    text, flags = _llm_article_text_and_flags(long_item)
    assert len(text) == MAX_ARTICLE_CHARS
    assert "article_truncated" in flags
