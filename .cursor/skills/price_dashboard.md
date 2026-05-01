# Price Dashboard Skill

When building comparison UI:

- Always show loading skeleton while fetching
- Highlight best price in green (#00C851)
- Highlight fastest delivery in blue (#007AFF)
- Show "Best Value" badge using composite score:
  score = (1/price * 0.5) + (1/delivery_days * 0.3) + (rating * 0.2)
- Always show "last updated" timestamp
- Add refresh button that invalidates React Query cache
