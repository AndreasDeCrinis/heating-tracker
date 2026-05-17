import threading
from datetime import date as real_date

import app as heating_app


def test_weather_history_update_runs_in_background(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_update():
        calls.append("updated")
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(heating_app, "_weather_update_thread", None)
    monkeypatch.setattr(heating_app, "update_weather_history_if_needed", fake_update)

    thread = heating_app.start_weather_history_update_async()

    assert thread is not None
    assert thread.daemon
    assert started.wait(timeout=1)
    assert thread.is_alive()

    release.set()
    thread.join(timeout=1)

    assert calls == ["updated"]
    assert not thread.is_alive()


def test_weather_history_update_skips_duplicate_background_run(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_update():
        calls.append("updated")
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(heating_app, "_weather_update_thread", None)
    monkeypatch.setattr(heating_app, "update_weather_history_if_needed", fake_update)

    first_thread = heating_app.start_weather_history_update_async()
    assert started.wait(timeout=1)

    second_thread = heating_app.start_weather_history_update_async()

    release.set()
    first_thread.join(timeout=1)

    assert second_thread is None
    assert calls == ["updated"]


def test_index_starts_weather_sync_asynchronously(monkeypatch):
    started = []

    monkeypatch.setattr(heating_app, "ensure_data_files", lambda: None)
    monkeypatch.setattr(heating_app, "start_weather_history_update_async", lambda: started.append(True))
    monkeypatch.setattr(
        heating_app,
        "update_weather_history_if_needed",
        lambda: (_ for _ in ()).throw(AssertionError("sync update should not run in index")),
    )
    monkeypatch.setattr(heating_app, "load_readings", lambda: [])
    monkeypatch.setattr(heating_app, "load_tariffs", lambda: {})
    monkeypatch.setattr(heating_app, "load_offsets", lambda: [])
    monkeypatch.setattr(heating_app, "load_grid_power", lambda: [])
    monkeypatch.setattr(heating_app, "load_weather_history", lambda: {})
    monkeypatch.setattr(heating_app, "render_template", lambda *args, **kwargs: "ok")

    heating_app.app.config.update(TESTING=True)
    with heating_app.app.test_client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "ok"
    assert started == [True]


def test_weather_history_backfills_missing_days(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    readings_csv = data_dir / "readings.csv"
    weather_csv = data_dir / "weather.csv"

    readings_csv.write_text(
        "date,meter_reading\n"
        "01.01.2024,100\n"
        "03.01.2024,120\n",
        encoding="utf-8",
    )
    weather_csv.write_text(
        "date,avg_temp_c,min_temp_c,max_temp_c\n"
        "2024-01-02,2.0,1.0,3.0\n",
        encoding="utf-8",
    )

    class FixedDate(real_date):
        @classmethod
        def today(cls):
            return cls(2024, 1, 3)

    fetch_calls = []

    def fake_fetch(start, end):
        fetch_calls.append((start, end))
        return {
            day: {"avg": float(day.day), "min": float(day.day) - 1, "max": float(day.day) + 1}
            for day in heating_app.daterange(start, end)
        }

    monkeypatch.setattr(heating_app, "date", FixedDate)
    monkeypatch.setattr(heating_app, "DATA_DIR", data_dir)
    monkeypatch.setattr(heating_app, "READINGS_CSV", readings_csv)
    monkeypatch.setattr(heating_app, "WEATHER_CSV", weather_csv)
    monkeypatch.setattr(heating_app, "fetch_historical_weather", fake_fetch)

    heating_app.update_weather_history_if_needed()

    history = heating_app.load_weather_history()
    assert fetch_calls == [
        (FixedDate(2024, 1, 1), FixedDate(2024, 1, 1)),
        (FixedDate(2024, 1, 3), FixedDate(2024, 1, 3)),
    ]
    assert set(history) == {
        FixedDate(2024, 1, 1),
        FixedDate(2024, 1, 2),
        FixedDate(2024, 1, 3),
    }
    assert history[FixedDate(2024, 1, 2)] == {"avg": 2.0, "min": 1.0, "max": 3.0}
