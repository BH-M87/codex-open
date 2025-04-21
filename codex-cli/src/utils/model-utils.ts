import { getBaseUrl, getApiKey } from "./config";
import OpenAI from "openai";

export const RECOMMENDED_MODELS: Array<string> = [
  "o4-mini",
  "o3",
  "deepseek-v3",
];

/**
 * Background model loader / cache.
 *
 * We start fetching the list of available models from OpenAI once the CLI
 * enters interactive mode.  The request is made exactly once during the
 * lifetime of the process and the results are cached for subsequent calls.
 */

async function fetchModels(provider: string): Promise<Array<string>> {
  // If the user has not configured an API key we cannot hit the network.
  if (!getApiKey(provider)) {
    throw new Error("No API key configured for provider: " + provider);
  }

  const baseURL = getBaseUrl(provider);
  try {
    const openai = new OpenAI({ apiKey: getApiKey(provider), baseURL });
    const list = await openai.models.list();
    const models: Array<string> = [];
    for await (const model of list as AsyncIterable<{ id?: string }>) {
      if (model && typeof model.id === "string") {
        let modelStr = model.id;
        // fix for gemini
        if (modelStr.startsWith("models/")) {
          modelStr = modelStr.replace("models/", "");
        }
        models.push(modelStr);
      }
    }

    return models.sort();
  } catch (error) {
    return [];
  }
}

export async function getAvailableModels(
  provider: string,
): Promise<Array<string>> {
  return fetchModels(provider.toLowerCase());
}

/**
 * Check if the provided model is in the list of recommended models.
 * This function no longer blocks any models, but returns whether the model
 * is in the recommended list for informational purposes.
 *
 * @param model The model name to check
 * @returns true for all models (no longer blocks any models)
 */
export async function isModelSupportedForResponses(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _model: string | undefined | null,
): Promise<boolean> {
  // Always return true - we no longer block any models
  return true;
}

/**
 * Check if a model is in the recommended list.
 * This is used to show warnings for non-recommended models.
 *
 * @param model The model name to check
 * @returns true if the model is recommended, false otherwise
 */
export function isRecommendedModel(model: string | undefined | null): boolean {
  if (typeof model !== "string" || model.trim() === "") {
    return true;
  }

  return RECOMMENDED_MODELS.includes(model.trim());
}
