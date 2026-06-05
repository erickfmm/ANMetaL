import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    globals: true,
    alias: {
      anmetal: path.resolve(__dirname, "dist/anmetal.esm.js"),
    },
  },
});
