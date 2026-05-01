# Authentication Skill

When building auth code:

- Always hash passwords with bcrypt (rounds=12)
- JWT access token expires in 24h
- Refresh token expires in 7 days
- Store refresh tokens in Redis (revocation support)
- Every protected route uses Depends(get_current_user)
- Return 401 with WWW-Authenticate header on failure
- Never return password hash in any response
