import { afterEach, describe, expect, it, vi } from "vitest";
import { stream } from "../src/api/antigravity.ts";
import { getModel } from "../src/compat.ts";
import type { Context, Model } from "../src/types.ts";

const context: Context = {
	messages: [{ role: "user", content: "hello", timestamp: Date.now() }],
};

function sseBody(events: unknown[]): ReadableStream<Uint8Array> {
	const text = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
	return new ReadableStream<Uint8Array>({
		start(controller) {
			controller.enqueue(new TextEncoder().encode(text));
			controller.close();
		},
	});
}

interface CapturedRequest {
	url: string;
	headers: Record<string, string>;
	body: {
		project?: string;
		request?: {
			tools?: Array<{
				functionDeclarations: Array<{ parametersJsonSchema?: unknown }>;
			}>;
			toolConfig?: { functionCallingConfig?: { mode?: string } };
		};
	};
}

function mockFetch(events: unknown[]): CapturedRequest[] {
	const captured: CapturedRequest[] = [];
	vi.stubGlobal(
		"fetch",
		vi.fn(async (url: string, init?: RequestInit): Promise<Response> => {
			captured.push({
				url,
				headers: (init?.headers ?? {}) as Record<string, string>,
				body: JSON.parse(String(init?.body)),
			});
			return new Response(sseBody(events), {
				status: 200,
				headers: { "Content-Type": "text/event-stream" },
			});
		}),
	);
	return captured;
}

afterEach(() => {
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});

describe("Antigravity API stream", () => {
	it("uses the project header (case-insensitive) for the payload project field", async () => {
		const captured = mockFetch([{ response: { candidates: [{ finishReason: "STOP" }] } }]);

		await stream(getModel("antigravity", "gemini-3.6-flash-high"), context, {
			apiKey: "test-token",
			headers: { "X-Antigravity-Project": "my-custom-project" },
		}).result();

		expect(captured).toHaveLength(1);
		expect(captured[0].url).toBe(
			"https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse",
		);
		expect(captured[0].body.project).toBe("my-custom-project");
		expect(captured[0].headers.Authorization).toBe("Bearer test-token");
	});

	it("defaults the project to aicode-consumers", async () => {
		const captured = mockFetch([{ response: { candidates: [{ finishReason: "STOP" }] } }]);

		await stream(getModel("antigravity", "gemini-3.6-flash-high"), context, {
			apiKey: "test-token",
		}).result();

		expect(captured[0].body.project).toBe("aicode-consumers");
	});

	it("carries thoughtSignature on tool calls and reports full usage", async () => {
		mockFetch([
			{
				response: {
					candidates: [
						{
							content: {
								parts: [{ thought: true, text: "thinking..." }],
							},
						},
					],
				},
			},
			{
				response: {
					candidates: [
						{
							content: {
								parts: [
									{
										functionCall: { name: "echo", args: { value: "hi" } },
										thoughtSignature: "opaque-signature",
									},
								],
							},
						},
					],
				},
			},
			{
				response: {
					candidates: [{ finishReason: "STOP" }],
					usageMetadata: {
						promptTokenCount: 110,
						candidatesTokenCount: 7,
						thoughtsTokenCount: 3,
						totalTokenCount: 120,
						cachedContentTokenCount: 10,
					},
				},
			},
		]);

		const result = await stream(getModel("antigravity", "gemini-3.6-flash-high"), context, {
			apiKey: "test-token",
		}).result();

		expect(result.stopReason).toBe("toolUse");
		expect(result.rawStopReason).toBe("STOP");

		const thinking = result.content.find((b) => b.type === "thinking");
		expect(thinking?.type === "thinking" && thinking.thinking).toBe("thinking...");

		const toolCall = result.content.find((b) => b.type === "toolCall");
		expect(toolCall).toMatchObject({
			type: "toolCall",
			name: "echo",
			arguments: { value: "hi" },
			thoughtSignature: "opaque-signature",
		});
		expect(typeof toolCall?.id).toBe("string");
		expect(toolCall?.id.length).toBeGreaterThan(0);

		expect(result.usage).toMatchObject({
			input: 100,
			output: 10,
			cacheRead: 10,
			reasoning: 3,
			totalTokens: 120,
		});
	});

	it("regenerates duplicate function call ids", async () => {
		mockFetch([
			{
				response: {
					candidates: [
						{
							content: {
								parts: [{ functionCall: { id: "call-1", name: "echo", args: { a: 1 } } }],
							},
						},
					],
				},
			},
			{
				response: {
					candidates: [
						{
							content: {
								parts: [{ functionCall: { id: "call-1", name: "echo", args: { a: 2 } } }],
							},
						},
					],
				},
			},
			{ response: { candidates: [{ finishReason: "STOP" }] } },
		]);

		const result = await stream(getModel("antigravity", "gemini-3.6-flash-high"), context, {
			apiKey: "test-token",
		}).result();

		const calls = result.content.filter((b) => b.type === "toolCall");
		expect(calls).toHaveLength(2);
		const ids = new Set(calls.map((c) => (c.type === "toolCall" ? c.id : "")));
		expect(ids.size).toBe(2);
	});

	it("fails closed for non-Gemini models", async () => {
		const model = {
			...getModel("antigravity", "gemini-3.6-flash-high"),
			id: "claude-sonnet-4-6",
		} as Model<"antigravity">;

		const result = await stream(model, context, { apiKey: "test-token" }).result();

		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("Unsupported Antigravity model: claude-sonnet-4-6");
	});

	it("uses parametersJsonSchema for Gemini models", async () => {
		const captured = mockFetch([{ response: { candidates: [{ finishReason: "STOP" }] } }]);

		await stream(
			getModel("antigravity", "gemini-3.6-flash-high"),
			{
				...context,
				tools: [{ name: "echo", description: "Echo a value", parameters: { type: "object", properties: {} } }],
			},
			{ apiKey: "test-token" },
		).result();

		const tools = captured[0].body.request?.tools;
		expect(tools?.[0].functionDeclarations[0].parametersJsonSchema).toBeDefined();
	});

	it("applies tool choice and request lifecycle hooks", async () => {
		const onResponse = vi.fn();
		const fetchFn = vi.fn(async (): Promise<Response> => {
			return new Response(sseBody([{ response: { candidates: [{ finishReason: "STOP" }] } }]), {
				status: 200,
				headers: { "X-Test": "seen" },
			});
		});

		await stream(
			getModel("antigravity", "gemini-3.6-flash-high"),
			{
				...context,
				tools: [{ name: "echo", description: "Echo a value", parameters: { type: "object", properties: {} } }],
			},
			{ apiKey: "test-token", fetch: fetchFn, onResponse, toolChoice: "none" },
		).result();

		const init = fetchFn.mock.calls[0]?.[1];
		const body = JSON.parse(String(init?.body)) as CapturedRequest["body"];
		expect(body.request?.toolConfig?.functionCallingConfig?.mode).toBe("NONE");
		expect(onResponse).toHaveBeenCalledWith(
			expect.objectContaining({ status: 200, headers: expect.objectContaining({ "x-test": "seen" }) }),
			expect.objectContaining({ provider: "antigravity" }),
		);
	});

	it("retries retryable HTTP responses", async () => {
		const fetchFn = vi
			.fn<() => Promise<Response>>()
			.mockResolvedValueOnce(new Response("rate limited", { status: 429, headers: { "retry-after-ms": "0" } }))
			.mockResolvedValueOnce(
				new Response(sseBody([{ response: { candidates: [{ finishReason: "STOP" }] } }]), { status: 200 }),
			);

		const result = await stream(getModel("antigravity", "gemini-3.6-flash-high"), context, {
			apiKey: "test-token",
			fetch: fetchFn,
			maxRetries: 1,
		}).result();

		expect(fetchFn).toHaveBeenCalledTimes(2);
		expect(result.stopReason).toBe("stop");
	});

	it("flushes a final SSE event split across CRLF chunks", async () => {
		const event = JSON.stringify({
			response: { candidates: [{ content: { parts: [{ text: "complete" }] }, finishReason: "STOP" }] },
		});
		const fetchFn = vi.fn(async (): Promise<Response> => {
			const body = new ReadableStream<Uint8Array>({
				start(controller) {
					controller.enqueue(new TextEncoder().encode(`data: ${event}\r`));
					controller.enqueue(new TextEncoder().encode("\n"));
					controller.close();
				},
			});
			return new Response(body, { status: 200 });
		});

		const result = await stream(getModel("antigravity", "gemini-3.6-flash-high"), context, {
			apiKey: "test-token",
			fetch: fetchFn,
		}).result();

		expect(result.content).toContainEqual(expect.objectContaining({ type: "text", text: "complete" }));
		expect(result.stopReason).toBe("stop");
	});
});
