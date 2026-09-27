/** @type {import('next').NextConfig} */
const path = require("path");

const nextConfig = {
  reactStrictMode: false,
  images: { unoptimized: true },

  // Proxy API calls to the Express gateway on Render (avoids CORS)
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${
          process.env.BACKEND_URL || "https://oceanembed-gateway.onrender.com"
        }/api/v1/:path*`,
      },
    ];
  },

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
