import typescript from "@rollup/plugin-typescript";
import terser from "@rollup/plugin-terser";

export default {
  input: "src/index.ts",
  output: [
    {
      file: "dist/anmetal.min.js",
      format: "umd",
      name: "ANMetaL",
      exports: "named",
      sourcemap: false,
    },
    {
      file: "dist/anmetal.esm.js",
      format: "esm",
      sourcemap: false,
    },
  ],
  plugins: [typescript({ tsconfig: "./tsconfig.json" }), terser()],
};
