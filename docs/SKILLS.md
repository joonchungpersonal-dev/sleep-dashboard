# Technologies & Skills Demonstrated

## Backend Development

### Python Async/Await
- Custom async HTTP client with `aiohttp`
- Proper session management and cleanup
- Concurrent API calls with rate limit awareness

### OAuth2 Implementation
- Password grant flow authentication
- Token expiration tracking and auto-refresh
- Secure credential handling via environment variables

### API Reverse Engineering
- Analyzed network traffic to discover new auth endpoints
- Identified OAuth2 client credentials from open-source forks
- Implemented working solution when official libraries failed

## Data Engineering

### Pandas Data Processing
- Complex aggregation (groupby, idxmax)
- Time series manipulation with timezone awareness
- Statistical calculations (correlation, standard deviation)

### Data Quality
- Outlier detection and filtering (HRV 10-150ms range)
- Multi-session deduplication per night
- Main sleep window filtering (6PM-6AM)

### ETL Pipeline
- Real-time API data ingestion
- DataFrame transformation and enrichment
- Caching with TTL for performance

## Frontend Development

### Streamlit
- Multi-section dashboard layout
- Real-time data refresh
- Session state management
- Custom theming (dark mode)

### Plotly Visualizations
- Interactive scatter plots with hover
- Step charts (hypnogram)
- Horizontal bar charts (sleep timeline)
- Dual-axis time series
- Correlation plots with regression lines

### UX Design
- Apple-style dark theme (#1C1C1E background)
- Color-coded metrics (green/red indicators)
- Responsive layout with columns
- Clear data hierarchy

## DevOps & Infrastructure

### Version Control
- Git commit best practices
- Meaningful commit messages
- Feature-based development

### Environment Management
- Python virtual environments
- Environment variable configuration
- .gitignore for sensitive data

### Cloud Ready (GCP)
- Cloud Run deployment scripts
- Cloud Functions for webhooks
- GCS for data storage
- IAP authentication

## Domain Knowledge

### Sleep Science
- Sleep stage classification (Deep, REM, Light, Awake)
- Circadian rhythm metrics (midpoint, variability)
- HRV as health indicator
- Sleep duration research thresholds

### Statistical Analysis
- Pearson correlation coefficient
- Standard deviation for variability
- Data filtering for accuracy
- Trend analysis with rolling averages

## Problem Solving

### Debugging Complex Issues
- Traced `pyeight` 400 errors to auth changes
- Identified HRV outliers affecting averages
- Fixed multi-session aggregation bugs
- Resolved timezone handling issues

### Performance Optimization
- Streamlit caching (10-min TTL)
- Efficient DataFrame operations
- Lazy loading of API data
- Rate limit backoff strategy

## Code Quality

### Clean Architecture
- Separation of concerns (ingestion/dashboard/config)
- Reusable chart functions
- Configuration externalization
- Type hints and docstrings

### Error Handling
- Graceful API failure handling
- User-friendly error messages
- Fallback values for missing data
- Retry logic with exponential backoff
