import Redis from "ioredis";

let client: Redis | null | undefined;

export function getRedis(): Redis | null {
  if (client !== undefined) return client;
  const url = process.env.REDIS_URL;
  if (!url) {
    client = null;
    return client;
  }
  try {
    client = new Redis(url, { maxRetriesPerRequest: 1, enableReadyCheck: false, lazyConnect: true });
    void client.connect().catch(() => {
      client = null;
    });
    return client;
  } catch {
    client = null;
    return client;
  }
}
