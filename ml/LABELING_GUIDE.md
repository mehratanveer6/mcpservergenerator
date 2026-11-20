# Risk labelling guide

Used for the gold evaluation set (`ml/data/gold_labels.csv`, 183 operations from 8
held-out providers: Stripe, Slack, Square, Plaid, Spotify, Asana, Mux, OpenAI).
The guide is aligned with the MCP tool annotation semantics: `readOnlyHint` for
operations that do not modify their environment and `destructiveHint` for updates
that are not purely additive.

| Label | Definition | Examples |
|---|---|---|
| `read` | No state change on the server. Safe to run without asking. | `GET /v1/customers`, `POST /v2/team-members/search`, `POST /identity/get`, `POST /transactions/enrich` |
| `write` | Creates or updates state in a recoverable way. | `POST /v1/customers`, `PUT /me/albums`, `POST /users.profile.set`, `POST /v2/subscriptions` |
| `destructive` | Deletes data, removes an association, revokes access, cancels, stops, archives, rejects, reverses or refunds. Hard or impossible to undo, or removes something a user relies on. | `DELETE /tasks/{id}`, `POST /chat.delete`, `POST /v1/charges/{id}/refund`, `POST /v1/payouts/{id}/reverse`, `POST /conversations.archive`, `POST /v1/disputes/{id}/close` |

Decision rules

1. Judge the effect on the remote system, not the HTTP method.
2. A POST that only computes or returns data (search, get, list, calculate, enrich) is `read`.
3. Creating an object is `write`, even when it starts a process (refresh, retry, verify).
4. Any operation whose main effect is removal, cancellation or reversal is `destructive`,
   including refunds created through a create-style endpoint.
5. Test-helper endpoints follow the same rules as the endpoint they simulate.

The training corpus is not hand-labelled: it is labelled by the rules in
`mcpforge/rules.py` (weak supervision). Only the gold set follows this guide.
