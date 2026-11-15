"""Calendar utilities, Indian public holidays, and synthetic events engine."""

from datetime import date, datetime

import holidays


class CalendarEngine:
    """Provides holiday, event, and weekend annotations for dates."""

    def __init__(self, country: str = "IN", years: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026, 2027)):
        self.country = country
        self.years = years
        try:
            self._holidays = holidays.country_holidays(country, years=years)
        except Exception:
            # Fallback gazetted Indian holidays if package fails
            self._holidays = {}
            for yr in years:
                self._holidays[date(yr, 1, 26)] = "Republic Day"
                self._holidays[date(yr, 8, 15)] = "Independence Day"
                self._holidays[date(yr, 10, 2)] = "Gandhi Jayanti"
                self._holidays[date(yr, 12, 25)] = "Christmas"
                self._holidays[date(yr, 1, 1)] = "New Year's Day"

        self._holiday_dates: set[date] = set(self._holidays.keys())
        self._sorted_holiday_dates = sorted(self._holiday_dates)

        # Synthetic recurring local events (e.g. annual tech summit, cultural festival)
        self._event_periods = [
            # Annual Tech Summit (Mid-February, Mon-Fri: corporate surge)
            (date(2023, 2, 13), date(2023, 2, 17)),
            (date(2024, 2, 12), date(2024, 2, 16)),
            (date(2025, 2, 10), date(2025, 2, 14)),
            (date(2026, 2, 9), date(2026, 2, 13)),
            # Cultural & Literature Festival (Early November: leisure surge)
            (date(2023, 11, 3), date(2023, 11, 7)),
            (date(2024, 11, 1), date(2024, 11, 5)),
            (date(2025, 11, 7), date(2025, 11, 11)),
            # Medical Congress (Late August: corporate surge)
            (date(2023, 8, 21), date(2023, 8, 25)),
            (date(2024, 8, 19), date(2024, 8, 23)),
            (date(2025, 8, 18), date(2025, 8, 22)),
        ]

    def _to_date(self, d: date | datetime | str) -> date:
        if isinstance(d, str):
            return datetime.strptime(d, "%Y-%m-%d").date()
        if isinstance(d, datetime):
            return d.date()
        return d

    def is_holiday(self, d: date | datetime | str) -> bool:
        """Return True if date is a public holiday."""
        dt = self._to_date(d)
        return dt in self._holiday_dates

    def holiday_name(self, d: date | datetime | str) -> str | None:
        """Return name of public holiday if present, else None."""
        dt = self._to_date(d)
        return self._holidays.get(dt, None)

    def is_weekend(self, d: date | datetime | str) -> bool:
        """Return True for Friday and Saturday nights (standard hotel leisure weekend)."""
        dt = self._to_date(d)
        return dt.weekday() in (4, 5)  # 4=Friday, 5=Saturday

    def is_event_period(self, d: date | datetime | str) -> bool:
        """Return True if date falls within a known local event period."""
        dt = self._to_date(d)
        for start, end in self._event_periods:
            if start <= dt <= end:
                return True
        return False

    def days_to_next_holiday(self, d: date | datetime | str) -> int:
        """Compute calendar days until the next upcoming holiday (capped at 60)."""
        dt = self._to_date(d)
        future = [h for h in self._sorted_holiday_dates if h >= dt]
        if future:
            return min((future[0] - dt).days, 60)
        return 60

    def days_since_last_holiday(self, d: date | datetime | str) -> int:
        """Compute calendar days since the most recent past holiday (capped at 60)."""
        dt = self._to_date(d)
        past = [h for h in self._sorted_holiday_dates if h <= dt]
        if past:
            return min((dt - past[-1]).days, 60)
        return 60
