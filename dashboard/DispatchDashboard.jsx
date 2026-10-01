import { useState, useEffect, useCallback } from "react";

// ── Capacity Tier Data ────────────────────────────────────────────────────────
const TIERS = [
  {
    id: 0, label: "Tier 0", name: "Local Hardware", color: "#00ff9d", bg: "#00ff9d12",
    description: "Your own machines — always free",
    providers: [
      { id:"rtx3090", name:"RTX 3090", note:"vLLM · 936 GB/s · CUDA", cost:"FREE", speed:"★★★★★", quality:"★★★☆☆", models:["qwen2.5-coder:7b","deepseek-coder-v2:16b"], status:"online", rpm:200 },
      { id:"macmini", name:"Mac Mini M4", note:"Ollama · 120 GB/s · always-on", cost:"FREE", speed:"★★★☆☆", quality:"★★★☆☆", models:["phi4-mini","qwen2.5:14b","nomic-embed"], status:"online", rpm:60 },
      { id:"macair",  name:"MacBook Air M5", note:"LM Studio MLX · 153 GB/s · mobile", cost:"FREE", speed:"★★★★☆", quality:"★★★☆☆", models:["phi4-mini (MLX)"], status:"online", rpm:30 },
    ],
  },
  {
    id: 1, label: "Tier 1", name: "Free API Quotas", color: "#60a5fa", bg: "#60a5fa12",
    description: "Free-tier APIs — rate limited, zero cost",
    providers: [
      { id:"groq",       name:"Groq",          note:"600 tok/s · 14.4k TPM/day · LPU", cost:"FREE", speed:"★★★★★", quality:"★★★★☆", models:["llama-3.3-70b","llama-3.1-8b"], status:"online", rpm:30 },
      { id:"openrouter", name:"OpenRouter",    note:"100+ free models", cost:"FREE", speed:"★★★☆☆", quality:"★★★☆☆", models:["mistral-7b:free","llama-3.1-8b:free"], status:"online", rpm:20 },
      { id:"gemini-free",name:"Gemini Free",   note:"1M context · OAuth free tier", cost:"FREE", speed:"★★★★☆", quality:"★★★★☆", models:["gemini-2.0-flash-lite"], status:"online", rpm:30 },
    ],
  },
  {
    id: 2, label: "Tier 2", name: "Budget API", color: "#f59e0b", bg: "#f59e0b12",
    description: "Cheap per-token — already in monthly budget",
    providers: [
      { id:"deepseek",   name:"DeepSeek",      note:"$0.27/M input — cheapest quality", cost:"$0.27/M", speed:"★★★☆☆", quality:"★★★★★", models:["deepseek-chat","deepseek-reasoner (R1)"], status:"online", rpm:60 },
      { id:"gemini-pay", name:"Gemini Flash",  note:"$0.075/M · 1M ctx · Google", cost:"$0.075/M", speed:"★★★★☆", quality:"★★★★☆", models:["gemini-2.0-flash"], status:"online", rpm:1000 },
      { id:"haiku",      name:"Claude Haiku",  note:"$0.80/M · Anthropic fast model", cost:"$0.80/M", speed:"★★★★☆", quality:"★★★★☆", models:["claude-haiku-4-5"], status:"online", rpm:50 },
      { id:"gpt4mini",   name:"GPT-4o Mini",   note:"$0.15/M · OpenAI budget", cost:"$0.15/M", speed:"★★★★☆", quality:"★★★★☆", models:["gpt-4o-mini"], status:"online", rpm:500 },
      { id:"kimi",       name:"Kimi 128k",     note:"$1.20/M · multilingual · long ctx", cost:"$1.20/M", speed:"★★★☆☆", quality:"★★★★☆", models:["moonshot-v1-128k"], status:"online", rpm:60 },
    ],
  },
  {
    id: 3, label: "Tier 3", name: "Premium API", color: "#e879f9", bg: "#e879f912",
    description: "Quality-critical tasks only — higher cost",
    providers: [
      { id:"sonnet",  name:"Claude Sonnet",  note:"$3/M · best balance · 200k ctx", cost:"$3/M",  speed:"★★★☆☆", quality:"★★★★★", models:["claude-sonnet-4-6"], status:"online", rpm:50 },
      { id:"gpt4o",   name:"GPT-4o",        note:"$2.50/M · OpenAI flagship", cost:"$2.50/M", speed:"★★★☆☆", quality:"★★★★★", models:["gpt-4o"], status:"online", rpm:500 },
      { id:"gem-pro", name:"Gemini Pro",    note:"$1.25/M · 1M ctx · Google best", cost:"$1.25/M", speed:"★★★☆☆", quality:"★★★★★", models:["gemini-2.5-pro"], status:"online", rpm:360 },
      { id:"opus",    name:"Claude Opus",   note:"$15/M · best quality · rare use", cost:"$15/M", speed:"★★☆☆☆", quality:"★★★★★", models:["claude-opus-4-6"], status:"online", rpm:20 },
    ],
  },
];

const TASK_ROUTES = {
  coding:       ["rtx3090","groq","deepseek","sonnet"],
  reasoning:    ["rtx3090","deepseek","groq","sonnet"],
  fast_chat:    ["macmini","macair","groq","gemini-pay"],
  long_context: ["gemini-free","gemini-pay","kimi","gem-pro"],
  multilingual: ["kimi","gemini-free","gemini-pay","gem-pro"],
  math:         ["deepseek","groq","sonnet","gpt4o"],
  quality:      ["sonnet","gpt4o","gem-pro","opus"],
  general:      ["macmini","openrouter","groq","deepseek"],
};

const MOCK_STATS = {
  monthly_spend: 3.42, monthly_limit: 50,
  daily_spend: 0.41,
  total_requests: 4821,
  cache_hit_rate: 0.38,
  requests_by_tier: [3104, 892, 701, 124],
  spend_by_provider: {deepseek:1.82, "gemini-pay":0.73, haiku:0.51, kimi:0.22, sonnet:0.14},
};

// ── Utils ─────────────────────────────────────────────────────────────────────
const fmtUsd = n => `$${(n||0).toFixed(2)}`;
const fmtPct = n => `${((n||0)*100).toFixed(1)}%`;
const fmt    = n => (n||0).toLocaleString();

// ── Components ─────────────────────────────────────────────────────────────────

function TierBadge({ tier }) {
  const t = TIERS[tier];
  return (
    <span style={{
      fontSize:9, padding:"2px 7px", borderRadius:3,
      background:t.bg, color:t.color,
      fontFamily:"monospace", textTransform:"uppercase", letterSpacing:"0.08em",
      border:`1px solid ${t.color}30`,
    }}>{t.label}</span>
  );
}

function ProviderRow({ p, tierColor, active }) {
  return (
    <div style={{
      display:"flex", alignItems:"center", gap:12,
      padding:"10px 14px", borderRadius:7,
      background: active ? `${tierColor}0d` : "#06060f",
      border:`1px solid ${active ? tierColor+"33" : "#0f0f1e"}`,
      transition:"all 0.3s",
    }}>
      <div style={{
        width:6, height:6, borderRadius:"50%",
        background: p.status === "online" ? tierColor : "#333",
        boxShadow: p.status === "online" ? `0 0 8px ${tierColor}` : "none",
        flexShrink:0,
      }} />
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:2 }}>
          <span style={{ fontWeight:700, color:"#e0e0f0", fontSize:12 }}>{p.name}</span>
          <span style={{ fontSize:9, color:"#444", fontFamily:"monospace" }}>{p.note}</span>
        </div>
        <div style={{ display:"flex", flexWrap:"wrap", gap:3 }}>
          {p.models.map(m => (
            <span key={m} style={{
              fontSize:8, padding:"1px 5px", borderRadius:3,
              background:"#0d0d1a", color:"#4a4a7a", fontFamily:"monospace",
              border:"1px solid #141428",
            }}>{m}</span>
          ))}
        </div>
      </div>
      <div style={{ textAlign:"right", flexShrink:0 }}>
        <div style={{ fontSize:12, fontWeight:700, color: p.cost==="FREE" ? "#00ff9d" : "#f59e0b", fontFamily:"monospace" }}>{p.cost}</div>
        <div style={{ fontSize:9, color:"#555", fontFamily:"monospace" }}>{p.rpm} RPM</div>
      </div>
      <div style={{ fontSize:10, color:"#555" }}>{p.quality}</div>
    </div>
  );
}

function RouteChain({ task, route }) {
  const taskColor = {
    coding:"#ff6b35", reasoning:"#c084fc", fast_chat:"#00ff9d",
    long_context:"#38bdf8", multilingual:"#fb923c", math:"#a3e635",
    quality:"#fbbf24", general:"#94a3b8",
  }[task] || "#888";

  return (
    <div style={{ marginBottom:12 }}>
      <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:6 }}>
        <span style={{
          fontSize:9, padding:"2px 9px", borderRadius:3, minWidth:88, textAlign:"center",
          background:`${taskColor}1a`, color:taskColor,
          fontFamily:"monospace", textTransform:"uppercase", letterSpacing:"0.08em",
          border:`1px solid ${taskColor}30`,
        }}>{task.replace("_"," ")}</span>
        <div style={{ flex:1, height:1, background:"#0f0f1e" }} />
      </div>
      <div style={{ display:"flex", alignItems:"center", gap:4, flexWrap:"wrap", paddingLeft:8 }}>
        {route.map((pid, i) => {
          const tier = TIERS.find(t => t.providers.some(p => p.id === pid));
          const prov = tier?.providers.find(p => p.id === pid);
          return (
            <div key={pid} style={{ display:"flex", alignItems:"center", gap:4 }}>
              {i > 0 && <span style={{ color:"#222", fontSize:12 }}>→</span>}
              <div style={{
                background:"#080816", border:`1px solid ${tier?.color||"#333"}22`,
                borderRadius:5, padding:"3px 9px",
                display:"flex", alignItems:"center", gap:5,
              }}>
                <span style={{ width:5, height:5, borderRadius:"50%", background:tier?.color||"#333", display:"inline-block", flexShrink:0 }} />
                <span style={{ fontSize:10, color:"#9090b0", fontFamily:"monospace" }}>{prov?.name||pid}</span>
                <span style={{ fontSize:8, color:prov?.cost==="FREE"?"#00ff9d":"#888", fontFamily:"monospace" }}>
                  {prov?.cost||"?"}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────────
export default function DispatchDashboard() {
  const [stats, setStats] = useState(MOCK_STATS);
  const [tab, setTab] = useState("tiers");
  const [testPrompt, setTestPrompt] = useState("");
  const [testResult, setTestResult] = useState(null);
  const [activeTiers, setActiveTiers] = useState([0,1,2]);  // Tier 3 gated

  useEffect(() => {
    const iv = setInterval(() => {
      setStats(prev => ({
        ...prev,
        total_requests: prev.total_requests + Math.floor(Math.random()*3),
        daily_spend: prev.daily_spend + (Math.random() > 0.85 ? 0.002 : 0),
      }));
    }, 3000);
    return () => clearInterval(iv);
  }, []);

  const budgetPct = (stats.monthly_spend / stats.monthly_limit) * 100;

  const classify = useCallback((p) => {
    const lower = p.toLowerCase();
    if (/code|function|def |debug|implement|script|algorithm/.test(lower)) return {task:"coding", route:TASK_ROUTES.coding};
    if (/reason|analyze|architecture|design|compare|tradeoff/.test(lower)) return {task:"reasoning", route:TASK_ROUTES.reasoning};
    if (/translate|spanish|french|chinese|japanese|arabic|persian/.test(lower)) return {task:"multilingual", route:TASK_ROUTES.multilingual};
    if (/summarize.*doc|analyze.*file|entire|whole codebase/.test(lower)) return {task:"long_context", route:TASK_ROUTES.long_context};
    if (/calculate|solve|equation|proof|derivative/.test(lower)) return {task:"math", route:TASK_ROUTES.math};
    if (/best possible|critical|thorough|production.?ready/.test(lower)) return {task:"quality", route:TASK_ROUTES.quality};
    if (p.length < 80 || /hi|hello|what is|quick/.test(lower)) return {task:"fast_chat", route:TASK_ROUTES.fast_chat};
    return {task:"general", route:TASK_ROUTES.general};
  }, []);

  const handleTest = useCallback(() => {
    if (!testPrompt.trim()) return;
    const {task, route} = classify(testPrompt);
    const firstAvail = route[0];
    const tier = TIERS.find(t => t.providers.some(p => p.id === firstAvail));
    const prov = tier?.providers.find(p => p.id === firstAvail);
    setTestResult({ task, route, firstAvail, tier:tier?.id, tierName:tier?.name, provName:prov?.name, cost:prov?.cost, tierColor:tier?.color });
  }, [testPrompt, classify]);

  const TABS = ["tiers","routing","cost","test"];

  return (
    <div style={{
      minHeight:"100vh", background:"#03030c", color:"#a0a0c0",
      fontFamily:"'JetBrains Mono','Fira Code',monospace", padding:"20px",
    }}>
      <style>{`
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-thumb{background:#141428;border-radius:2px}
        input{background:#06060f;border:1px solid #141428;border-radius:6px;
              padding:9px 13px;color:#c0c0dc;font-family:monospace;font-size:12px;width:100%}
        input:focus{outline:none;border-color:#00ff9d44}
        button{cursor:pointer;font-family:monospace}
      `}</style>

      {/* Header */}
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:20 }}>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <div style={{
            width:34, height:34, borderRadius:7, display:"flex", alignItems:"center", justifyContent:"center",
            background:"linear-gradient(135deg,#00ff9d,#60a5fa)", fontSize:18,
            boxShadow:"0 0 24px #00ff9d44",
          }}>⌬</div>
          <div>
            <div style={{ fontSize:17, fontWeight:800, color:"#f0f0ff", letterSpacing:"-0.02em" }}>DISPATCH</div>
            <div style={{ fontSize:8, color:"#333", letterSpacing:"0.2em", textTransform:"uppercase" }}>
              4-Tier LLM Router · LiteLLM Engine · Local + Cloud
            </div>
          </div>
        </div>

        {/* Budget gauge */}
        <div style={{ textAlign:"right" }}>
          <div style={{ fontSize:9, color:"#333", marginBottom:3 }}>MONTHLY BUDGET</div>
          <div style={{ fontSize:13, fontWeight:700, color: budgetPct>80?"#ef4444":budgetPct>50?"#f59e0b":"#00ff9d", fontFamily:"monospace" }}>
            {fmtUsd(stats.monthly_spend)} / {fmtUsd(stats.monthly_limit)}
          </div>
          <div style={{ width:160, height:4, background:"#0f0f1e", borderRadius:2, marginTop:4 }}>
            <div style={{
              height:"100%", borderRadius:2, transition:"width 0.5s",
              width:`${Math.min(budgetPct,100)}%`,
              background: budgetPct>80?"#ef4444":budgetPct>50?"#f59e0b":"#00ff9d",
              boxShadow:`0 0 8px ${budgetPct>80?"#ef4444":budgetPct>50?"#f59e0b":"#00ff9d"}88`,
            }} />
          </div>
        </div>
      </div>

      {/* Top stats */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:8, marginBottom:18 }}>
        {[
          {label:"Requests", val:fmt(stats.total_requests), color:"#00ff9d"},
          {label:"Cache Hits", val:fmtPct(stats.cache_hit_rate), color:"#60a5fa"},
          {label:"Daily Spend", val:fmtUsd(stats.daily_spend), color:"#f59e0b"},
          {label:"Monthly Spend", val:fmtUsd(stats.monthly_spend), color:"#e879f9"},
        ].map(({label,val,color}) => (
          <div key={label} style={{ background:"#06060f", border:"1px solid #0f0f1e", borderRadius:7, padding:"11px 14px" }}>
            <div style={{ fontSize:8, color:"#333", textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:3 }}>{label}</div>
            <div style={{ fontSize:18, fontWeight:700, color, fontFamily:"monospace" }}>{val}</div>
          </div>
        ))}
      </div>

      {/* Tier availability bar */}
      <div style={{ display:"flex", gap:6, marginBottom:18 }}>
        {TIERS.map(t => {
          const enabled = t.id <= 2 || budgetPct < 90;
          const reqs = stats.requests_by_tier[t.id] || 0;
          return (
            <div key={t.id} style={{
              flex:1, background:"#06060f", border:`1px solid ${enabled ? t.color+"33" : "#0f0f1e"}`,
              borderRadius:7, padding:"10px 12px",
              opacity: enabled ? 1 : 0.4,
            }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
                <TierBadge tier={t.id} />
                <span style={{ fontSize:9, color: enabled ? t.color : "#333", fontFamily:"monospace" }}>
                  {enabled ? "OPEN" : "GATED"}
                </span>
              </div>
              <div style={{ fontSize:11, color:"#c0c0d8", fontWeight:600, marginBottom:2 }}>{t.name}</div>
              <div style={{ fontSize:9, color:"#444" }}>{fmt(reqs)} requests</div>
            </div>
          );
        })}
      </div>

      {/* Tabs */}
      <div style={{ display:"flex", gap:1, marginBottom:16, borderBottom:"1px solid #0f0f1e" }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab===t ? "#06060f" : "transparent",
            border:"none", borderBottom: tab===t ? "2px solid #00ff9d" : "2px solid transparent",
            color: tab===t ? "#00ff9d" : "#333",
            padding:"7px 16px", fontSize:9, textTransform:"uppercase", letterSpacing:"0.1em",
            transition:"all 0.2s",
          }}>{t}</button>
        ))}
      </div>

      {/* ── Tiers Tab ── */}
      {tab === "tiers" && (
        <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
          {TIERS.map(tier => (
            <div key={tier.id} style={{
              background:"#05050e", border:`1px solid ${tier.color}22`,
              borderLeft:`3px solid ${tier.color}`, borderRadius:8, padding:16,
            }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:12 }}>
                <div style={{ display:"flex", alignItems:"center", gap:10 }}>
                  <TierBadge tier={tier.id} />
                  <span style={{ color:"#e0e0f0", fontWeight:700, fontSize:13 }}>{tier.name}</span>
                  <span style={{ fontSize:10, color:"#444" }}>{tier.description}</span>
                </div>
                <span style={{ fontSize:10, color: tier.id<=2 ? "#00ff9d" : budgetPct<90?"#e879f9":"#ef4444",
                  fontFamily:"monospace" }}>
                  {tier.id<=2 ? "ALWAYS AVAILABLE" : budgetPct<90 ? "GATED — quality tasks" : "BUDGET LIMIT"}
                </span>
              </div>
              <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
                {tier.providers.map(p => <ProviderRow key={p.id} p={p} tierColor={tier.color} active={tier.id<=2} />)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Routing Tab ── */}
      {tab === "routing" && (
        <div style={{ display:"grid", gridTemplateColumns:"3fr 2fr", gap:16 }}>
          <div>
            <div style={{ fontSize:9, color:"#333", marginBottom:14, textTransform:"uppercase", letterSpacing:"0.1em" }}>
              Task → Priority Chain (left = first tried, right = fallback)
            </div>
            {Object.entries(TASK_ROUTES).map(([task, route]) => (
              <RouteChain key={task} task={task} route={route} />
            ))}
          </div>
          <div style={{ background:"#06060f", border:"1px solid #0f0f1e", borderRadius:9, padding:18 }}>
            <div style={{ fontSize:9, color:"#333", marginBottom:14, textTransform:"uppercase" }}>Routing Logic</div>
            {[
              ["Task classify", "Regex + context length"],
              ["Budget gate", "Blocks tier 2/3 at spend limits"],
              ["Strategy", "cost_first | speed_first | quality_first | local_only"],
              ["Provider fail", "LiteLLM: cooldown + auto-retry"],
              ["Rate limit (429)", "LiteLLM: cooldown → next deployment"],
              ["Context overflow", "LiteLLM: context_window_fallbacks"],
              ["All fail", "Fallback to local-always-on"],
            ].map(([k,v]) => (
              <div key={k} style={{ display:"flex", justifyContent:"space-between", padding:"8px 0", borderBottom:"1px solid #0a0a18" }}>
                <span style={{ fontSize:10, color:"#555" }}>{k}</span>
                <span style={{ fontSize:10, color:"#8080a0", textAlign:"right", maxWidth:180 }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Cost Tab ── */}
      {tab === "cost" && (
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
          <div style={{ background:"#06060f", border:"1px solid #0f0f1e", borderRadius:9, padding:18 }}>
            <div style={{ fontSize:9, color:"#333", marginBottom:14, textTransform:"uppercase" }}>Spend by Provider (month)</div>
            {Object.entries(stats.spend_by_provider).sort(([,a],[,b])=>b-a).map(([pid,amt]) => {
              const tier = TIERS.find(t => t.providers.some(p => p.id===pid));
              const prov = tier?.providers.find(p => p.id===pid);
              const max = Math.max(...Object.values(stats.spend_by_provider));
              return (
                <div key={pid} style={{ marginBottom:12 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                    <div style={{ display:"flex", alignItems:"center", gap:6 }}>
                      <span style={{ width:6, height:6, borderRadius:"50%", background:tier?.color||"#666", display:"inline-block" }} />
                      <span style={{ fontSize:11, color:"#9090b0" }}>{prov?.name||pid}</span>
                      <TierBadge tier={tier?.id||0} />
                    </div>
                    <span style={{ fontSize:11, color:"#f59e0b", fontFamily:"monospace" }}>{fmtUsd(amt)}</span>
                  </div>
                  <div style={{ height:3, background:"#0a0a18", borderRadius:2 }}>
                    <div style={{ height:"100%", width:`${(amt/max)*100}%`, background:tier?.color||"#666", borderRadius:2, transition:"width 0.5s" }} />
                  </div>
                </div>
              );
            })}
            <div style={{ marginTop:16, padding:"10px 0", borderTop:"1px solid #0f0f1e", display:"flex", justifyContent:"space-between" }}>
              <span style={{ fontSize:10, color:"#444" }}>Tier 0 (local)</span>
              <span style={{ fontSize:11, color:"#00ff9d", fontFamily:"monospace" }}>FREE · {fmt(stats.requests_by_tier[0])} requests</span>
            </div>
          </div>

          <div style={{ background:"#06060f", border:"1px solid #0f0f1e", borderRadius:9, padding:18 }}>
            <div style={{ fontSize:9, color:"#333", marginBottom:14, textTransform:"uppercase" }}>Cost Efficiency</div>
            {[
              ["Requests served free", `${((stats.requests_by_tier[0]+stats.requests_by_tier[1])/stats.total_requests*100).toFixed(0)}%`,"#00ff9d"],
              ["Avg cost/request", `${(stats.monthly_spend/stats.total_requests*100).toFixed(4)}¢`,"#f59e0b"],
              ["vs all-cloud equiv.", fmtUsd(stats.total_requests * 0.002),"#ef4444"],
              ["Monthly savings", fmtUsd(stats.total_requests * 0.002 - stats.monthly_spend),"#00ff9d"],
            ].map(([k,v,c]) => (
              <div key={k} style={{ display:"flex", justifyContent:"space-between", padding:"11px 0", borderBottom:"1px solid #0a0a18" }}>
                <span style={{ fontSize:11, color:"#555" }}>{k}</span>
                <span style={{ fontSize:13, fontWeight:700, color:c, fontFamily:"monospace" }}>{v}</span>
              </div>
            ))}
            <div style={{ marginTop:14, fontSize:9, color:"#222", lineHeight:2 }}>
              Local-first routing routes {((stats.requests_by_tier[0])/stats.total_requests*100).toFixed(0)}% of<br/>
              requests to your hardware at zero marginal cost.
            </div>
          </div>
        </div>
      )}

      {/* ── Test Tab ── */}
      {tab === "test" && (
        <div style={{ maxWidth:640 }}>
          <div style={{ fontSize:9, color:"#333", marginBottom:12, textTransform:"uppercase", letterSpacing:"0.1em" }}>
            Route Preview — See exactly which tier/provider handles your prompt
          </div>
          <div style={{ display:"flex", gap:8, marginBottom:16 }}>
            <input value={testPrompt} onChange={e => setTestPrompt(e.target.value)}
              onKeyDown={e => e.key==="Enter" && handleTest()}
              placeholder="Type any prompt to preview routing..." />
            <button onClick={handleTest} style={{
              background:"#00ff9d1a", border:"1px solid #00ff9d33",
              color:"#00ff9d", borderRadius:6, padding:"9px 16px", fontSize:11, whiteSpace:"nowrap",
            }}>Preview →</button>
          </div>

          {testResult && (
            <div style={{
              background:"#06060f", border:`1px solid ${testResult.tierColor}33`,
              borderLeft:`3px solid ${testResult.tierColor}`, borderRadius:8, padding:18, marginBottom:16,
            }}>
              <div style={{ display:"flex", gap:8, marginBottom:14, flexWrap:"wrap" }}>
                <span style={{ fontSize:9, padding:"2px 8px", borderRadius:3, background:"#0f0f1e", color:"#9090b0", fontFamily:"monospace" }}>
                  task: {testResult.task}
                </span>
                <TierBadge tier={testResult.tier} />
              </div>
              {[
                ["Selected Provider", testResult.provName, testResult.tierColor],
                ["Tier", `${testResult.tier} — ${testResult.tierName}`, testResult.tierColor],
                ["Cost", testResult.cost, testResult.cost==="FREE"?"#00ff9d":"#f59e0b"],
              ].map(([k,v,c]) => (
                <div key={k} style={{ display:"flex", justifyContent:"space-between", padding:"9px 0", borderBottom:"1px solid #0a0a18" }}>
                  <span style={{ fontSize:11, color:"#444" }}>{k}</span>
                  <span style={{ fontSize:12, fontWeight:700, color:c }}>{v}</span>
                </div>
              ))}
              <div style={{ marginTop:14 }}>
                <div style={{ fontSize:9, color:"#333", marginBottom:8 }}>FULL PRIORITY CHAIN</div>
                <div style={{ display:"flex", flexWrap:"wrap", gap:4 }}>
                  {testResult.route.map((pid, i) => {
                    const tier = TIERS.find(t => t.providers.some(p => p.id===pid));
                    const prov = tier?.providers.find(p => p.id===pid);
                    return (
                      <div key={pid} style={{ display:"flex", alignItems:"center", gap:3 }}>
                        {i>0 && <span style={{ color:"#1a1a2e", fontSize:10 }}>→</span>}
                        <span style={{
                          fontSize:9, padding:"2px 7px", borderRadius:3,
                          background: i===0 ? `${tier?.color}1a` : "#0a0a18",
                          color: i===0 ? tier?.color : "#444",
                          fontFamily:"monospace", border:`1px solid ${i===0?tier?.color+"33":"#0f0f1e"}`,
                        }}>{prov?.name||pid}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
          <div style={{ fontSize:9, color:"#1a1a2e", lineHeight:2.2 }}>
            "Write a Python function" → coding → RTX 3090 (FREE)<br/>
            "Quick summary please" → fast_chat → Mac Mini phi4-mini (FREE)<br/>
            "Analyze this 100k doc" → long_context → Gemini Flash free ($0)<br/>
            "Translate to Persian" → multilingual → Kimi 128k ($1.20/M)<br/>
            "Most thorough analysis" → quality → Claude Sonnet ($3/M)
          </div>
        </div>
      )}

      <div style={{ marginTop:22, paddingTop:12, borderTop:"1px solid #0a0a18", display:"flex", justifyContent:"space-between" }}>
        <span style={{ fontSize:8, color:"#1a1a2e" }}>
          DISPATCH v3 · :8080 API · LiteLLM :4000 UI · OpenAI-compatible
        </span>
        <span style={{ fontSize:8, color:"#1a1a2e" }}>{new Date().toLocaleTimeString()}</span>
      </div>
    </div>
  );
}
