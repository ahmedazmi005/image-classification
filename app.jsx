import { useState } from "react";
import "./app.css";

const VISION_API_URL = "http://127.0.0.1:8001/classify_image";
const CHAT_API_URL = "http://127.0.0.1:8001/chat";

function App() {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const addMessage = (msg) =>
    setMessages((prev) => [...prev, { id: prev.length + 1, ...msg }]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    const trimmed = newMessage.trim();
    if (!trimmed) return;
    addMessage({ sender: "user", text: trimmed });
    setNewMessage("");

    try {
      const res = await fetch(CHAT_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });
      if (!res.ok) {
        addMessage({
          sender: "bot",
          text: "Chat backend error talking to AnythingLLM.",
        });
        return;
      }
      const data = await res.json();
      const reply = data.reply || "[No reply text returned]";
      addMessage({ sender: "bot", text: reply });
    } catch (err) {
      console.error(err);
      addMessage({
        sender: "bot",
        text: "Failed to reach the local chat backend.",
      });
    }
  };

  const handleImageChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    addMessage({ sender: "user", image: imageUrl });

    const formData = new FormData();
    formData.append("image", file);

    setIsUploading(true);
    try {
      const res = await fetch(VISION_API_URL, { method: "POST", body: formData });
      if (!res.ok) {
        addMessage({
          sender: "bot",
          text: "Sorry, I couldn't classify that image.",
        });
        return;
      }
      const data = await res.json();
      const confidence =
        typeof data.confidence === "number"
          ? ` (${(data.confidence * 100).toFixed(1)}% confidence)`
          : "";
      addMessage({
        sender: "bot",
        text: `Prediction: ${data.result}${confidence}`,
      });
    } catch (err) {
      console.error(err);
      addMessage({
        sender: "bot",
        text: "Error talking to the local model. Is the backend running?",
      });
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-log">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`chat-bubble ${message.sender}`}
          >
            {message.text && <div>{message.text}</div>}
            {message.image && (
              <img
                src={message.image}
                alt="Uploaded"
                className="uploaded-image"
              />
            )}
          </div>
        ))}
        {messages.length === 0 && (
          <div className="chat-bubble bot">
            Upload a skin image with the paperclip or send a message to start.
          </div>
        )}
        {isUploading && (
          <div className="chat-bubble bot">
            Classifying image with your local model…
          </div>
        )}
      </div>

      <form onSubmit={handleSendMessage} className="chat-form">
        <input
          type="text"
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Type a message..."
        />
        <button type="submit">Send</button>
        <label htmlFor="file-upload" className="attach-button">
          📎 Attach
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleImageChange}
        />
      </form>
    </div>
  );
}

export default App;