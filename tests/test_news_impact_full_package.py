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
    assert not Path("configs/news_impact.json").exists()
    assert Path("data/news_impact/watchlist.example.csv").exists()
    assert Path("data/news_impact/company_master.example.csv").exists()


def test_news_impact_runtime_examples_keep_openai_default_and_gemma_option():
    openai_example = Path("configs/news_impact.example.json").read_text(encoding="utf-8")
    gemma_example = Path("configs/news_impact.gemma.example.json").read_text(encoding="utf-8")

    assert '"llm_provider": "openai"' in openai_example
    assert '"llm_model": "gpt-5-mini"' in openai_example
    assert "OPENAI_API_KEY" not in openai_example
    assert '"llm_provider": "llama_cpp"' in gemma_example
    assert '"llm_model": "gemma-4-26b-a4b"' in gemma_example


def test_news_impact_llm_prompt_asset_is_vendored():
    from pathlib import Path

    # impact_judge.build_system_prompt()가 런타임에 읽는 필수 자산. 머지 시 누락 방지.
    assert Path("docs/NEWS_IMPACT_LLM_PROMPT.md").exists()
