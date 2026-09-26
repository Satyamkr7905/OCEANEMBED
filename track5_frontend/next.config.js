/** @type {import('next').NextConfig} */
const path = require("path");

const nextConfig = {
  reactStrictMode: false,
  images: { unoptimized: true },
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "mapbox-gl": "maplibre-gl",
      "plotly.js/dist/plotly": path.resolve(__dirname, "node_modules/plotly.js/dist/plotly.min.js"),
    };
    return config;
  },
};

module.exports = nextConfig;
