from pathlib import Path


def test_news_impact_full_package_importable_with_internal_imports():
    from news_impact.pipeline import DailyPipelineInputs, run_daily_pipeline
    from news_impact.schema import ImpactEvent, NewsItem
    from news_impact.stock_factors.classifier import classify_factors

    assert DailyPipelineInputs.__name__ == "DailyPipelineInputs"
    assert callable(run_daily_pipeline)
    assert ImpactEvent.__name__ == "ImpactEvent"
    assert NewsItem.__name__ == "NewsItem"
    assert callable(classify_factors)


def test_pyproject_packages_include_migrated_news_impact_package():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"news_impact"' in pyproject
    assert '"news_impact.stock_factors"' in pyproject
    assert 'stock-news-impact = "news_impact.run:main"' in pyproject


def test_news_impact_runtime_examples_are_migrated_without_private_config():
    assert Path("configs/news_impact.example.json").exists()
    assert not Path("configs/news_impact.json").exists()
    assert Path("data/news_impact/watchlist.example.csv").exists()
    assert Path("data/news_impact/company_master.example.csv").exists()
