#!/usr/bin/env node
/**
 * Local embedding server using node-llama-cpp and GGUF models.
 * 
 * Modes:
 *   Single:     node _embed_node.mjs <model.gguf> "text"
 *   Streaming:  node _embed_node.mjs <model.gguf> --stream
 *               Then send one text per line on stdin, get one JSON array per line on stdout.
 *               Send "EXIT" to quit.
 */

import { getLlama } from "node-llama-cpp";
import { createInterface } from "readline";
import { resolve } from "path";

const modelPath = process.argv[2];
const isStream = process.argv[3] === "--stream";

if (!modelPath) {
  console.error("Usage: node _embed_node.mjs <model.gguf> [text | --stream]");
  process.exit(1);
}

async function main() {
  const llama = await getLlama();
  const model = await llama.loadModel({ modelPath: resolve(modelPath) });
  const context = await model.createEmbeddingContext();

  if (isStream) {
    // Keep model loaded, process line by line
    process.stdout.write("READY\n");
    
    const rl = createInterface({ input: process.stdin, terminal: false });
    
    for await (const line of rl) {
      if (line.trim() === "EXIT") break;
      if (!line.trim()) {
        process.stdout.write("[]\n");
        continue;
      }
      try {
        const embedding = await context.getEmbeddingFor(line);
        process.stdout.write(JSON.stringify(Array.from(embedding.vector)) + "\n");
      } catch (err) {
        process.stderr.write(`Error: ${err.message}\n`);
        process.stdout.write("[]\n");
      }
    }
    
    process.exit(0);
  } else {
    // Single text mode
    const text = process.argv.slice(3).join(" ");
    if (!text) {
      console.error("No text provided");
      process.exit(1);
    }
    const embedding = await context.getEmbeddingFor(text);
    console.log(JSON.stringify(Array.from(embedding.vector)));
    process.exit(0);
  }
}

main().catch(err => {
  console.error(err.message);
  process.exit(1);
});
