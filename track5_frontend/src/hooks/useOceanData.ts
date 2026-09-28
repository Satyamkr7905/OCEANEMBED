"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchProfile, fetchRaster, pingGateway } from "@/lib/api";
import type { DepthM, MapLayerId, OceanProfile, RasterGrid } from "@/types/ocean";

export const DEFAULT_DATE = "2020-05-18";
export const DEFAULT_STATION = { lat: 16.4, lon: 87.2 };

export function useOceanData(date: string, depth: DepthM, layer: MapLayerId, selected: { lat: number; lon: number } | null) {
  const [online, setOnline] = useState(false);
  const [isWaking, setIsWaking] = useState(false);
  const [profile, setProfile] = useState<OceanProfile | null>(null);
  const [raster, setRaster] = useState<RasterGrid | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Background keep-alive and ping check
  useEffect(() => {
    let alive = true;
    const checkStatus = async () => {
      const ok = await pingGateway();
      if (alive) {
        setOnline(ok);
        if (ok) setIsWaking(false);
      }
    };
    void checkStatus();
    const id = window.setInterval(checkStatus, 12000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  const loadProfile = useCallback(async (lat: number, lon: number) => {
    setLoading(true);
    setError(null);

    // If request takes >3.5s, trigger cold-start waking banner
    const wakingTimer = setTimeout(() => setIsWaking(true), 3500);

    try {
      const data = await fetchProfile(lat, lon, date);
      clearTimeout(wakingTimer);
      setProfile(data);
      setError(null);
      setOnline(true);
      setIsWaking(false);
    } catch {
      clearTimeout(wakingTimer);
      setOnline(false);
      setIsWaking(false);
      setError("Sounding unavailable. The inference cloud server did not respond. Retrying background connection...");
    } finally {
      setLoading(false);
    }
  }, [date]);

  useEffect(() => {
    if (selected) void loadProfile(selected.lat, selected.lon);
  }, [selected, loadProfile]);

  useEffect(() => {
    let alive = true;
    const wakingTimer = setTimeout(() => setIsWaking(true), 3500);

    fetchRaster(layer, date, depth)
      .then((grid) => {
        clearTimeout(wakingTimer);
        if (!alive) return;
        setRaster(grid);
        setOnline(true);
        setIsWaking(false);
      })
      .catch(() => {
        clearTimeout(wakingTimer);
        if (!alive) return;
        setOnline(false);
      });

    return () => {
      alive = false;
      clearTimeout(wakingTimer);
    };
  }, [layer, date, depth]);

  return { online, isWaking, loading, profile, raster, error, reload: loadProfile };
}

