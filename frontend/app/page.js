"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  UploadCloud,
  FileText,
  Trash2,
  Loader2,
  Sparkles,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Search,
  BookOpen,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Menu,
  X
} from "lucide-react";

// Backend API base URL. Set NEXT_PUBLIC_API_URL in your environment
// (.env.local for dev, Vercel project settings for production) to your
// deployed backend's URL, e.g. https://malaika871-nexusai.hf.space
// Falls back to localhost for local development if not set.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  // State
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // { type: 'success'|'error', message: '' }
  const [isDragActive, setIsDragActive] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedSources, setExpandedSources] = useState({});

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch documents on load
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    setDocsLoading(true);
    try {
      const res = await fetch(`${API_URL}/documents`);
      const data = await res.json();
      setDocuments(data || []);
    } catch (err) {
      console.error("Error fetching documents:", err);
    } finally {
      setDocsLoading(false);
    }
  };

  const handleUpload = async (files) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();
      setUploadStatus({
        type: "success",
        message: data.indexed_files === 1
          ? "Document uploaded and ready to use."
          : `${data.indexed_files} documents uploaded and ready to use.`,
      });
      fetchDocuments();
    } catch (err) {
      console.error(err);
      setUploadStatus({
        type: "error",
        message: "Failed to upload and index documents. Please try again.",
      });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (docName) => {
    try {
      const res = await fetch(`${API_URL}/documents/${encodeURIComponent(docName)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setDocuments(prev => prev.filter(d => d !== docName));
      }
    } catch (err) {
      console.error("Error deleting document:", err);
    }
  };

  const askQuestion = async (queryText = question) => {
    const activeQuery = queryText.trim();
    if (!activeQuery) return;

    setQuestion("");
    setLoading(true);

    const userMsgId = Date.now().toString();
    const botMsgId = (Date.now() + 1).toString();

    // Add user message
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", text: activeQuery },
    ]);

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: activeQuery }),
      });

      if (!res.ok) throw new Error("API call failed");

      if (!res.body) {
        throw new Error("No response stream");
      }

      // Initialize empty bot message in list
      setMessages((prev) => [
        ...prev,
        { id: botMsgId, role: "bot", text: "", sources: [], streaming: true },
      ]);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = "";

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        buffer += decoder.decode(value, { stream: !done });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            try {
              const jsonStr = trimmed.slice(6);
              const data = JSON.parse(jsonStr);
              if (data.sources) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === botMsgId
                      ? { ...msg, sources: data.sources, contextType: data.context_type || msg.contextType }
                      : msg
                  )
                );
              }
              if (data.token) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === botMsgId
                      ? { ...msg, text: msg.text + data.token }
                      : msg
                  )
                );
              }
            } catch (err) {
              console.error("Error parsing SSE chunk:", err);
            }
          }
        }
      }

      // Done streaming
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === botMsgId ? { ...msg, streaming: false } : msg
        )
      );

    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: botMsgId,
          role: "bot",
          text: "Sorry, I encountered an error communicating with the backend. Make sure the backend server is running.",
          sources: [],
          error: true
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Drag & drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUpload(e.dataTransfer.files);
    }
  };

  const toggleSources = (msgId) => {
    setExpandedSources((prev) => ({
      ...prev,
      [msgId]: !prev[msgId]
    }));
  };

  const suggestions = [
    "What is the self-attention mechanism?",
    "Explain the key architectural differences in Transformers.",
    "How does RAG improve document retrieval accuracy?",
    "Analyze the optimization tricks discussed in the papers."
  ];

  return (
    <div className="flex h-screen bg-[#07070d] text-slate-100 overflow-hidden font-sans relative">
      
      {/* Background Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-purple-900/10 blur-[120px] pointer-events-none" />
      <div className="absolute top-[40%] right-[20%] w-[30%] h-[30%] rounded-full bg-emerald-950/10 blur-[120px] pointer-events-none" />

      {/* Main Container */}
      <div className="flex w-full h-full z-10 p-4 gap-4 relative">
        
        {/* Mobile menu toggle */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="absolute top-6 left-6 md:hidden z-50 p-2 rounded-lg bg-white/5 border border-white/10 text-slate-300 hover:text-white"
        >
          {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
        </button>

        {/* Sidebar Panel */}
        <AnimatePresence initial={false}>
          {sidebarOpen && (
            <motion.div
              initial={{ x: -320, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -320, opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="fixed inset-y-4 left-4 z-40 w-80 md:relative md:inset-0 flex flex-col gap-4"
            >
              <div className="flex-1 flex flex-col rounded-2xl backdrop-blur-md bg-slate-950/40 border border-white/5 p-5 overflow-hidden shadow-2xl">
                
                {/* Header */}
                <div className="flex items-center justify-between pb-4 border-b border-white/5">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 shadow-[0_0_15px_rgba(59,130,246,0.5)]">
                      <BookOpen size={20} className="text-white" />
                    </div>
                    <div>
                      <h1 className="font-bold text-lg bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                        NexusAI
                      </h1>
                      <p className="text-xs text-slate-500 font-mono">AI Research Partner</p>
                    </div>
                  </div>
                </div>

                {/* Drag and Drop Zone */}
                <div className="mt-5">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <UploadCloud size={14} /> Index Documents
                  </h3>
                  
                  <div
                    onDragEnter={handleDrag}
                    onDragOver={handleDrag}
                    onDragLeave={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={`relative group cursor-pointer border-2 border-dashed rounded-xl p-6 transition-all duration-300 text-center flex flex-col items-center justify-center ${
                      isDragActive
                        ? "border-blue-500 bg-blue-500/10"
                        : "border-white/10 hover:border-white/20 bg-white/2"
                    }`}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      className="hidden"
                      accept=".pdf,.docx,.txt"
                      onChange={(e) => handleUpload(e.target.files)}
                    />
                    
                    {uploading ? (
                      <div className="flex flex-col items-center gap-2">
                        <Loader2 className="animate-spin text-blue-500" size={32} />
                        <p className="text-sm font-medium text-slate-300">Processing & Indexing...</p>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-2">
                        <UploadCloud size={32} className="text-slate-400 group-hover:text-blue-400 transition-colors" />
                        <p className="text-xs text-slate-300">
                          <span className="font-semibold text-blue-400">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-[10px] text-slate-500">PDF, DOCX, TXT up to 25MB</p>
                      </div>
                    )}
                  </div>

                  {/* Upload Status Alert */}
                  {uploadStatus && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`mt-2 p-3 rounded-lg flex gap-2 text-xs items-start ${
                        uploadStatus.type === "success"
                          ? "bg-emerald-950/30 text-emerald-300 border border-emerald-500/20"
                          : "bg-rose-950/30 text-rose-300 border border-rose-500/20"
                      }`}
                    >
                    </motion.div>
                  )}
                </div>

                {/* Document List */}
                <div className="flex-1 flex flex-col min-h-0 mt-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                      <FileText size={14} /> Knowledge Library
                    </h3>
                    <button
                      onClick={fetchDocuments}
                      className="p-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
                      title="Refresh library"
                    >
                      <RefreshCw size={14} className={docsLoading ? "animate-spin text-blue-400" : ""} />
                    </button>
                  </div>

                  <div className="flex-1 overflow-y-auto pr-1 space-y-2 custom-scrollbar">
                    {docsLoading && documents.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-8 gap-2 text-slate-500">
                        <Loader2 className="animate-spin text-slate-400" size={20} />
                        <span className="text-xs">Fetching Knowledge base...</span>
                      </div>
                    ) : documents.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-12 px-4 rounded-xl border border-white/5 bg-white/2 text-slate-500 text-center">
                        <BookOpen size={28} className="mb-2 text-slate-600" />
                        <p className="text-xs font-medium text-slate-400">Library is empty</p>
                        <p className="text-[10px] text-slate-600 mt-1">Upload research documents above to enable RAG answering.</p>
                      </div>
                    ) : (
                      <AnimatePresence initial={false}>
                        {documents.map((docName) => (
                          <motion.div
                            key={docName}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                            className="group flex items-center justify-between p-3 rounded-xl bg-white/2 hover:bg-white/5 border border-white/5 hover:border-white/10 transition-all duration-200"
                          >
                            <div className="flex items-center gap-2.5 min-w-0">
                              <FileText size={16} className="text-indigo-400 shrink-0" />
                              <span className="text-xs font-medium truncate text-slate-300 group-hover:text-white transition-colors">
                                {docName}
                              </span>
                            </div>
                            <button
                              onClick={() => handleDelete(docName)}
                              className="p-1 rounded-md opacity-0 group-hover:opacity-100 hover:bg-rose-500/10 text-slate-500 hover:text-rose-400 transition-all"
                              title="Delete document"
                            >
                              <Trash2 size={13} />
                            </button>
                          </motion.div>
                        ))}
                      </AnimatePresence>
                    )}
                  </div>
                </div>

              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Chat Area Panel */}
        <div className="flex-1 flex flex-col rounded-2xl backdrop-blur-md bg-slate-950/40 border border-white/5 overflow-hidden shadow-2xl relative">
          
          {/* Chat Header */}
          <div className="flex items-center justify-between p-4 border-b border-white/5 md:pl-6 pl-16">
            <div className="flex items-center gap-2.5">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <div>
                <h2 className="text-sm font-semibold text-white">Assistant Chatbot</h2>
                <p className="text-[10px] text-slate-500 font-mono">Qwen 2.5 7B Instruct Enabled</p>
              </div>
            </div>
            
            {/* Show Sidebar button if collapsed */}
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-slate-300 hover:text-white hover:bg-white/10 transition-all"
              >
                <BookOpen size={14} /> Open Library
              </button>
            )}
          </div>

          {/* Conversation viewport */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center max-w-xl mx-auto text-center space-y-6">
                
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.1, duration: 0.4 }}
                  className="p-4 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400"
                >
                  <Sparkles size={40} className="animate-pulse" />
                </motion.div>
                
                <div>
                  <h2 className="text-2xl font-bold text-white bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                    How can I assist your research?
                  </h2>
                  <p className="text-sm text-slate-400 mt-2 max-w-md mx-auto">
                    Upload papers in the library, and ask questions. I will construct factual answers backed by sources in your indexed documents.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full pt-4">
                  {suggestions.map((sug, i) => (
                    <motion.button
                      key={i}
                      onClick={() => askQuestion(sug)}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.1 + 0.2 }}
                      className="p-3 text-left rounded-xl bg-white/2 hover:bg-indigo-500/5 hover:border-indigo-500/30 border border-white/5 transition-all text-xs font-medium text-slate-300 hover:text-white group relative overflow-hidden"
                    >
                      <span>{sug}</span>
                      <div className="absolute right-3 bottom-3 opacity-0 group-hover:opacity-100 text-indigo-400 transition-opacity">
                        <Send size={12} />
                      </div>
                    </motion.button>
                  ))}
                </div>

              </div>
            ) : (
              <div className="space-y-6 max-w-4xl mx-auto">
                <AnimatePresence initial={false}>
                  {messages.map((msg) => {
                    const isUser = msg.role === "user";
                    const isExpanded = expandedSources[msg.id];
                    return (
                      <motion.div
                        key={msg.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                        className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[85%] rounded-2xl p-4 shadow-xl border ${
                            isUser
                              ? "bg-gradient-to-br from-blue-600/20 to-indigo-600/20 border-blue-500/30 text-slate-100 rounded-tr-none"
                              : msg.error
                              ? "bg-rose-950/20 border-rose-500/30 text-rose-300 rounded-tl-none"
                              : "bg-slate-900/30 border-white/5 text-slate-200 rounded-tl-none"
                          }`}
                        >
                          
                          {/* Sender label */}
                          <div className="flex items-center gap-2 mb-1.5">
                            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500">
                              {isUser ? "You" : "Research Assistant"}
                            </span>
                            {!isUser && msg.streaming && (
                              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-ping" />
                            )}
                          </div>

                          {/* Message Body */}
                          <p className="text-sm leading-relaxed whitespace-pre-wrap font-sans break-words selection:bg-blue-500/30">
                            {msg.text || (msg.streaming && <span className="inline-block w-1.5 h-4 bg-blue-500 animate-pulse ml-0.5" />)}
                            {!isUser && msg.streaming && msg.text && (
                              <span className="inline-block w-1.5 h-4 bg-blue-500 animate-pulse ml-0.5" />
                            )}
                          </p>

                          {/* Sources Accordion */}
                          {!isUser && msg.sources && msg.sources.length > 0 && (
                            <div className="mt-4 pt-3 border-t border-white/5">
                              <button
                                onClick={() => toggleSources(msg.id)}
                                className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 font-semibold transition-colors"
                              >
                                {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                <span>Sources ({msg.sources.length})</span>
                              </button>

                              <AnimatePresence>
                                {isExpanded && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.2 }}
                                    className="overflow-hidden mt-2.5 space-y-2"
                                  >
                                    {msg.sources.map((src, idx) => (
                                      <div
                                        key={idx}
                                        className="p-2.5 rounded-lg bg-white/2 border border-white/5 text-[11px] flex justify-between items-center"
                                      >
                                        {src.url ? (
                                          <a
                                            href={src.url}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="flex items-center gap-2 truncate text-slate-300 hover:text-indigo-300 font-medium"
                                          >
                                            <Search size={12} className="text-indigo-400 shrink-0" />
                                            <span className="truncate">{src.source}</span>
                                          </a>
                                        ) : (
                                          <div className="flex items-center gap-2 truncate">
                                            <FileText size={12} className="text-slate-500 shrink-0" />
                                            <span className="text-slate-300 truncate font-medium">{src.source}</span>
                                          </div>
                                        )}
                                        {src.url ? (
                                          <span className="shrink-0 text-indigo-300 px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20 text-[10px] font-mono">
                                            Web
                                          </span>
                                        ) : src.page != null ? (
                                          <span className="shrink-0 text-slate-500 px-2 py-0.5 rounded bg-white/5 border border-white/5 text-[10px] font-mono">
                                            Page {src.page}
                                          </span>
                                        ) : null}
                                      </div>
                                    ))}
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </div>
                          )}

                        </div>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
                <div ref={chatEndRef} />
              </div>
            )}
          </div>

          {/* Floating Glass Input Bar */}
          <div className="p-4 border-t border-white/5 bg-slate-950/20">
            <div className="max-w-4xl mx-auto flex items-center gap-3 relative">
              
              <div className="relative flex-1 flex items-center">
                <input
                  className="w-full text-sm bg-white/2 border border-white/10 hover:border-white/15 focus:border-blue-500/50 rounded-xl py-3.5 pl-4 pr-12 text-slate-200 placeholder-slate-500 focus:outline-none transition-all shadow-[inset_0_2px_4px_0_rgba(0,0,0,0.4)] backdrop-blur-md"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      askQuestion();
                    }
                  }}
                  placeholder="Ask a question about your indexed papers..."
                  disabled={loading}
                />
                
                <button
                  onClick={() => askQuestion()}
                  disabled={loading || !question.trim()}
                  className={`absolute right-2.5 p-2 rounded-lg transition-all duration-300 ${
                    question.trim() && !loading
                      ? "bg-gradient-to-tr from-blue-600 to-indigo-600 hover:shadow-[0_0_15px_rgba(59,130,246,0.4)] text-white"
                      : "text-slate-600 cursor-not-allowed"
                  }`}
                >
                  {loading ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : (
                    <Send size={16} />
                  )}
                </button>

              </div>
            </div>
            <p className="text-[10px] text-center text-slate-600 mt-2">
              Note: Answers are retrieved based strictly on chunks matching your question.
            </p>
          </div>

      </div>

    </div>

  </div>
  );
}
