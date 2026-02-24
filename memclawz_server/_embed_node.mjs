// Generates embeddings using QMD's local GGUF model (embeddinggemma-300M)
// Usage: node _embed_node.mjs "text to embed"
// Output: JSON array of floats (768-dim)
import path from "path";
import os from "os";

const qmdLlama = "/opt/homebrew/lib/node_modules/@tobilu/qmd/node_modules/node-llama-cpp/dist/index.js";
const { getLlama } = await import(qmdLlama);

const text = process.argv[2] || "";
const modelPath = path.join(os.homedir(), ".cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf");

const llama = await getLlama({ logLevel: "fatal" });
const model = await llama.loadModel({ modelPath });
const ctx = await model.createEmbeddingContext();
const embedding = await ctx.getEmbeddingFor(text);
console.log(JSON.stringify(Array.from(embedding.vector)));
await ctx.dispose();
await model.dispose();
process.exit(0);
