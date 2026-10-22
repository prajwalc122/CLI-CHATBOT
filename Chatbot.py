"""CLI AI chatbot powered by Llama 3.2 running locally through Ollama."""

import sys
from datetime import datetime

import ollama

MODEL = "llama3.2"
SYSTEM_PROMPT = "You are a helpful, friendly AI assistant running in a terminal. Keep answers clear and concise."

HELP_TEXT = """Commands:
  /help   show this help
  /clear  clear the conversation memory
  /save   save the chat to a text file
  /exit   quit (Ctrl+C also works)"""


def new_history():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_chat(history):
    name = f"chat_{datetime.now():%Y%m%d_%H%M%S}.txt"
    with open(name, "w", encoding="utf-8") as f:
        for msg in history:
            if msg["role"] != "system":
                f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")
    print(f"Saved to {name}")


def stream_reply(history):
    """Send the history to the model and print the reply as it streams in."""
    reply = ""
    for chunk in ollama.chat(model=MODEL, messages=history, stream=True):
        piece = chunk["message"]["content"]
        reply += piece
        print(piece, end="", flush=True)
    print()
    return reply


def main():
    history = new_history()
    print(f"Chatting with {MODEL}. Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/exit":
                print("Bye!")
                break
            elif cmd == "/clear":
                history = new_history()
                print("Conversation cleared.")
            elif cmd == "/save":
                save_chat(history)
            elif cmd == "/help":
                print(HELP_TEXT)
            else:
                print("Unknown command. Type /help.")
            continue

        history.append({"role": "user", "content": user_input})
        print("AI: ", end="", flush=True)
        try:
            reply = stream_reply(history)
        except ollama.ResponseError as e:
            history.pop()
            print(f"\nModel error: {e.error}")
            if e.status_code == 404:
                print(f"Run this first:  ollama pull {MODEL}")
            continue
        except KeyboardInterrupt:
            history.pop()
            print("\n(interrupted)")
            continue
        except Exception:
            history.pop()
            print("\nCould not reach Ollama. Make sure it is running (open the Ollama app or run: ollama serve).")
            continue

        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    sys.exit(main())
