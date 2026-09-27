FROM node:20-alpine AS build
WORKDIR /src
COPY track4_backend/express_gateway/package.json track4_backend/express_gateway/package-lock.json* ./
RUN npm install
COPY track4_backend/express_gateway/tsconfig.json ./
COPY track4_backend/express_gateway/src ./src
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY track4_backend/express_gateway/package.json ./
RUN npm install --omit=dev
COPY --from=build /src/dist ./dist
EXPOSE 8080
CMD ["node", "dist/index.js"]

