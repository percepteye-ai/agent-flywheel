# In the dashboard

Every click path these docs mention is written once, on this page, and every other page links here. Sign in at [launch.percepteye.ai](https://launch.percepteye.ai); signing in is free.

## Get your key

**New mission**, in the sidebar → **Your AI agent** (selected by default) → under **How should we get your agent's conversations?**, choose **Add our SDK to your agent's code** → name the agent → **Create agent and get my key**.

The name is the agent id your code reports as. The key (`pefw_…`) is shown once, and it belongs to that one agent: a key made for another agent is refused. Copy it, and the control-plane URL shown with it, into your agent's environment as `PERCEPTEYE_API_KEY` and `PERCEPTEYE_CONTROL_PLANE_URL`.

The status panel beside it shows when the agent's first call arrives, then asks for [permission](#allow-us-to-learn-from-its-conversations) and offers **Start improving**.

Only an organization admin can create an agent and its key. A key expires after 180 days.

## Replace a lost key

**Agents** → your agent → **Get a new key**. The new key is shown once. The old one keeps working until it expires or you revoke it: **Settings** → **Agent keys** → **Revoke**.

## Send us its OpenTelemetry traces

**New mission** → **Your AI agent** → **Also send us the traces your agent already exports** → name the agent → **Create agent and get my key**.

The key is shown once, at the top. Below it, choose **I run an OpenTelemetry Collector** or **My agent exports directly**, and copy what it shows, the agent's endpoint included. The status panel shows when its first traces arrive, then offers **Start improving**.

## Connect your observability provider

**New mission** → **Your AI agent** → **Connect your observability provider**. We read your agent's traces from the observability provider they already go to; nothing changes in your agent.

## Upload a trace file

**New mission** → **Your AI agent** → **Upload a trace file**: an OpenTelemetry export of your agent's traces.

## Allow us to learn from its conversations

During setup, one checkbox: **Let us learn from its conversations, including the message text**. Until it is on, none of your agent's conversations are kept.

It switches on the agent's two permissions, which its page shows separately:

- **Send us its conversations**: the steps of each conversation, which tools your agent called and what it decided.
- **Include the message text**: the words of each message, and what the model saw.

Learning needs both. You can switch them off at any time on the agent's page, and your agent notices without a restart. Only an organization admin can change them.

## See what arrived

Your agent's page, its **Agent** tab: **What it has sent us to learn from** counts the conversation turns the SDK sent and the spans pushed over OpenTelemetry, and how many of each we keep but do not learn from, and why.

## Start improving

**Start improving**, at the end of setup or on your agent's page. It needs the permission on. If the round is paid from your credits, it shows the most it can cost before it starts. Run it again whenever new conversations have arrived.

## Run practice tasks

Your agent's page → **Advanced** → **Generate tasks and run them**, or **Generate, run and improve**. We write the tasks from your agent's own tools and system prompt; your agent works through them while it runs in training mode, `PERCEPTEYE_AGENT_MODE=training`, which you set explicitly (unset means production). Practice runs also need **Allow practice runs** turned on for the agent here.

## Approve the winner in Decisions

**Decisions**, in the sidebar. A new version comes here only after it has beaten your agent's current best on conversations it has never seen. Approve it or reject it; nothing else decides.

## Try a new version on one agent

Your agent's page, its **Agent** tab → **Try the candidate**. **Move back to champion** undoes it.

## Choose what we keep

**Settings** → **Rollout retention**. Off, the default, deletes nothing. On asks us to delete content older than the number of days you set; the card says whether that deletion is running for your organization yet. Only an organization admin can change it.
