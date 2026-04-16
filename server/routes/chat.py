"""WebSocket chat endpoint and HTTP chat fallback."""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from server.agent.coach import handle_chat, handle_chat_agentic
from server.models.schemas import ChatRequest, ChatResponse, ChatMessage, ChatRole
from server.llm.ollama import ollama_client
from server.llm.external import external_client
from server.agent.prompts import build_context_prompt, build_messages
from server.agent.tools import get_tool_definitions
from server.quiz_client.client import quiz_client

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── HTTP Chat (simple request/response) ─────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response from the study coach."""
    return await handle_chat(request)


@router.post("/chat/agentic", response_model=ChatResponse)
async def chat_agentic(request: ChatRequest):
    """Send a message and get an agentic response with tool-use and actions."""
    return await handle_chat_agentic(request)


# ─── WebSocket Chat (real-time streaming with agentic support) ───────────────


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with streaming responses.

    Client sends: {"user_id": "...", "message": "...", "history": [...]}
    Server sends: {"type": "token", "content": "..."} for each token
    Server sends: {"type": "action", "action": "...", "params": {...}, "label": "..."}
    Server sends: {"type": "done", "weaknesses": [...]} when complete
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            payload = json.loads(data)

            user_id = payload.get("user_id", "")
            user_message = payload.get("message", "")
            history = payload.get("history", [])

            if not user_id or not user_message:
                await websocket.send_text(
                    json.dumps({"type": "error", "content": "Missing user_id or message"})
                )
                continue

            # Fetch quiz data
            try:
                quiz_history = await quiz_client.get_player_history(user_id)
                quiz_data = [h.model_dump() for h in quiz_history] if quiz_history else []
            except Exception as e:
                logger.error(f"Failed to fetch quiz history: {e}")
                quiz_data = []

            # Weakness analysis
            weakness_data = None
            if quiz_data:
                try:
                    from server.learning.weakness import analyze_weaknesses

                    weakness_report = await analyze_weaknesses(user_id, quiz_data)
                    weakness_data = weakness_report.model_dump() if weakness_report else None
                except Exception as e:
                    logger.error(f"Weakness analysis failed: {e}")

            # Build context & messages (agentic mode)
            context = build_context_prompt(
                quiz_history=quiz_data,
                weakness_report=weakness_data,
            )
            messages = build_messages(user_message, context, history, agentic=True)

            # Agentic tool-use loop
            try:
                from server.agent.coach import _get_llm_client, MAX_TOOL_ROUNDS
                from server.agent.tool_executor import execute_tool
                from server.models.schemas import LLMResponse

                llm_client = await _get_llm_client()
                if not llm_client:
                    await websocket.send_text(
                        json.dumps({
                            "type": "error",
                            "content": "No LLM available. Configure an API key or start Ollama.",
                        })
                    )
                    continue

                tools = get_tool_definitions()
                pending_actions = []
                final_content = None

                for round_num in range(MAX_TOOL_ROUNDS):
                    llm_response: LLMResponse = await llm_client.chat_with_tools(
                        messages, tools
                    )

                    if llm_response.tool_calls:
                        logger.info(
                            f"WS Round {round_num + 1}: tool calls: "
                            f"{[tc.name for tc in llm_response.tool_calls]}"
                        )

                        # Build assistant message with tool calls
                        messages.append({
                            "role": "assistant",
                            "content": llm_response.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": str(tc.arguments),
                                    },
                                }
                                for tc in llm_response.tool_calls
                            ],
                        })

                        # Execute tools and send actions to frontend
                        for tc in llm_response.tool_calls:
                            result_text, action = await execute_tool(
                                tc, user_id, weakness_data
                            )

                            if action:
                                pending_actions.append(action)
                                # Send action to frontend immediately
                                await websocket.send_text(
                                    json.dumps({
                                        "type": "action",
                                        "action": action.action,
                                        "params": action.params,
                                        "label": action.label,
                                    })
                                )

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_text,
                            })
                    else:
                        # Final text response — stream it
                        final_content = llm_response.content
                        break

                # Stream the final response
                if final_content:
                    # Stream character-by-character for a natural feel
                    # (We already have the full response from the tool-use call)
                    # Send in chunks for smooth streaming
                    chunk_size = 4
                    for i in range(0, len(final_content), chunk_size):
                        chunk = final_content[i : i + chunk_size]
                        await websocket.send_text(
                            json.dumps({"type": "token", "content": chunk})
                        )
                else:
                    # No tool calls happened — use regular streaming
                    if await external_client.is_available():
                        async for token in external_client.chat_stream(messages):
                            await websocket.send_text(
                                json.dumps({"type": "token", "content": token})
                            )
                    elif await ollama_client.is_available():
                        async for token in ollama_client.chat_stream(messages):
                            await websocket.send_text(
                                json.dumps({"type": "token", "content": token})
                            )


                # Send done signal
                weaknesses = weakness_data.get("weakest_categories") if weakness_data else None
                await websocket.send_text(
                    json.dumps({"type": "done", "weaknesses": weaknesses})
                )

            except Exception as e:
                logger.error(f"Streaming failed: {e}", exc_info=True)
                await websocket.send_text(
                    json.dumps({"type": "error", "content": "Failed to generate response."})
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

