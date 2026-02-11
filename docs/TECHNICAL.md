# Technical Implementation Details

## Eight Sleep API Reverse Engineering

### The Problem
The official `pyeight` library stopped working in late 2024 when Eight Sleep deprecated their session token authentication. The library would fail with:
```
400 Bad Request: invalid_grant
```

### The Solution
Through API analysis, I discovered Eight Sleep migrated to OAuth2 password grant flow:

```python
# Old (broken) approach
POST https://client-api.8slp.net/v1/login
{"email": "...", "password": "..."}
# Returns session token - NO LONGER WORKS FOR DATA ACCESS

# New (working) approach
POST https://auth-api.8slp.net/v1/tokens
{
    "client_id": "<CLIENT_ID>",
    "client_secret": "<CLIENT_SECRET>",
    "grant_type": "password",
    "username": "...",
    "password": "..."
}
# Returns OAuth2 access_token - WORKS
```

Client credentials are sourced from environment variables (defaults from lukas-clarke/pyEight).

### Implementation
```python
class EightSleepClient:
    async def login(self, retries: int = 3) -> bool:
        auth_data = {
            "client_id": KNOWN_CLIENT_ID,
            "client_secret": KNOWN_CLIENT_SECRET,
            "grant_type": "password",
            "username": self.email,
            "password": self.password,
        }

        async with self._session.post(AUTH_URL, data=auth_data) as resp:
            if resp.status == 200:
                data = await resp.json()
                self._access_token = data.get("access_token")
                self._user_id = data.get("userId")
                # Token expires in ~20 hours
                expires_in = data.get("expires_in", 72000)
                self._token_expires = datetime.now() + timedelta(seconds=expires_in - 120)
```

## HRV Data Quality Issue

### The Problem
Raw HRV values from the API showed unrealistic readings:
```
HRV range: 22.5 - 1338.6 ms
HRV mean: 327.1 ms
```

Normal HRV (RMSSD) is typically 20-100ms for adults, rarely exceeding 150ms even for athletes.

### Analysis
The API returns raw R-R interval variability data with significant outliers. Examining the distribution:
```
Mean: 327.1 ms
Median: 261.1 ms
Filtered mean (10-200ms): 68.2 ms  <- Realistic!
25th percentile: 78.8 ms
```

### Solution
Filter HRV values to physiologically plausible range before averaging:
```python
def avg_timeseries(key, min_val=None, max_val=None):
    data = timeseries.get(key, [])
    values = [v[1] for v in data if v[1] is not None]
    if min_val is not None or max_val is not None:
        values = [v for v in values
                  if (min_val is None or v >= min_val)
                  and (max_val is None or v <= max_val)]
    return sum(values) / len(values) if values else None

# Usage
"hrv_avg": avg_timeseries("hrv", min_val=10, max_val=150)
```

## Sleep Session Aggregation

### The Problem
Eight Sleep records multiple "sessions" per night:
- Main sleep (7+ hours)
- Brief awakenings recorded as separate sessions
- Naps during the day

Simply averaging all sessions gives wrong metrics:
```
Session 1: 12:08 PM - 1.1h (nap)
Session 2: 09:28 AM - 1.2h (morning doze)
Session 3: 09:26 PM - 7.2h (main sleep)
Average: 3.2h  <- WRONG!
```

### Solution
1. **Filter to main sleep window** (6PM - 6AM start times)
2. **Group by night_date**
3. **Keep longest session per night** for timing metrics
4. **Sum durations** for total sleep time

```python
def is_main_sleep(row):
    ts = row.get("session_start")
    local_ts = ts.astimezone(local_tz)
    hour = local_ts.hour
    return hour >= 18 or hour < 6  # 6PM to 6AM

main_df = df[df.apply(is_main_sleep, axis=1)]
main_df = main_df.loc[main_df.groupby("night_date")["duration_hours"].idxmax()]
```

## Hypnogram Visualization

### Implementation
Sleep stages are stored as a sequence with durations:
```json
{"stages": [
    {"stage": "awake", "duration": 2970},
    {"stage": "light", "duration": 660},
    {"stage": "deep", "duration": 1260},
    ...
]}
```

Converted to cumulative time positions for plotting:
```python
stage_sequence = []
cumulative_mins = 0
for stage in stages:
    stage_sequence.append({
        "stage": stage_type,
        "start_min": cumulative_mins,
        "end_min": cumulative_mins + duration_mins,
    })
    cumulative_mins += duration_mins
```

Rendered as horizontal colored segments with a step-style connecting line:
```python
# Connecting line (background)
fig.add_trace(go.Scatter(
    x=line_x, y=line_y,
    line=dict(color="rgba(255,255,255,0.4)", width=2, shape="hv"),
))

# Colored segments (foreground)
for stage in stages:
    fig.add_trace(go.Scatter(
        x=[stage["start_min"], stage["end_min"]],
        y=[stage_map[stage_type], stage_map[stage_type]],
        line=dict(color=stage_colors[stage_type], width=6),
    ))
```

## Rate Limit Handling

The Eight Sleep API enforces rate limits (429 responses). Implemented exponential backoff:

```python
for attempt in range(retries):
    async with self._session.get(url, headers=headers) as resp:
        if resp.status == 200:
            return await resp.json()
        elif resp.status == 429:
            wait_time = 2 ** attempt * 5  # 5, 10, 20 seconds
            await asyncio.sleep(wait_time)
            continue
        elif resp.status == 401:
            # Token expired, refresh and retry
            await self.login()
            continue
```

## Correlation Analysis

Computed correlations across all numeric metrics to find relationships:

```python
hrv_corrs = []
for col in available_columns:
    corr = df['hrv_avg'].corr(df[col])
    hrv_corrs.append((col, corr))
hrv_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
```

Results:
- Duration ↔ REM: r = 0.88 (strong positive)
- HRV ↔ Heart Rate: r = 0.43
- HRV ↔ Awake Minutes: r = -0.39
- Duration ↔ Deep: varies

Visualized with scatter plots including regression lines and r-values.
