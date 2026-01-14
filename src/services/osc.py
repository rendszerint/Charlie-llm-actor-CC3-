import re
import math
import time
import threading
from collections import Counter
from typing import Optional, List, Dict
from pathlib import Path
from pythonosc.udp_client import SimpleUDPClient

class OscService:
    def __init__(self, ip: str = "127.0.0.1", port: int = 12000, log_file: Optional[Path] = None):
        self._client = SimpleUDPClient(ip, port)
        self._log_file = log_file
        
        # Counter state
        self._counter: float = 12.0
        self._min_counter: float = 12.0
        self._max_counter: float = 124.0
        self._increment: float = 0.01 # User set to 0.1
        self._idle_timeout: float = 60.0
        self._last_activity: float = 0.0
        
        # Dashboard State
        self._last_text: str = ""
        self._last_actions: List[str] = []
        
        # Threading
        self._running = True
        self._thread = threading.Thread(target=self._counter_loop, daemon=True)
        self._thread.start()

        # Word frequency tracking for progressive filtering
        self._word_counts: Counter = Counter()
        self._total_turns = 0
        self._retention_rate: float = 1.0
        self._min_retention: float = 0.1
        self._retention_decay: float = 0.05

    def stop(self):
        """Stops the background thread."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def report_activity(self):
        """Updates the last activity timestamp, keeping the counter alive."""
        self._last_activity = time.time()

    def _counter_loop(self):
        """Background loop to manage counter state."""
        while self._running:
            now = time.time()
            updated = False
            
            if now - self._last_activity < self._idle_timeout:
                # Active state: increment counter
                if self._counter < self._max_counter:
                    self._counter += self._increment
                    # Clamp to max
                    if self._counter > self._max_counter:
                        self._counter = self._max_counter
                    updated = True
            else:
                # Idle state: reset counter
                if self._counter != self._min_counter:
                    self._counter = self._min_counter
                    updated = True
            
            if updated:
                self.send_data()
            
            # Sleep interval controls speed (0.1 increment per 0.1s = 1.0 per second)
            time.sleep(0.1)

    def _update_word_stats(self, text: str):
        # normalize and count words
        words = re.findall(r'\b\w+\b', text.lower())
        self._word_counts.update(words)
        self._total_turns += 1
        
        # Decay retention rate
        self._retention_rate = max(
            self._min_retention, 
            self._retention_rate - self._retention_decay
        )

    def _filter_text(self, text: str) -> List[str]:
        words = re.findall(r'\b\w+\b', text) # get words with original casing
        if not words:
            return []

        # Calculate importance (frequency) for each word
        num_words = len(words)
        keep_count = math.ceil(num_words * self._retention_rate)
        
        if keep_count >= num_words:
            return words

        # Score words by their global frequency
        word_scores = []
        for i, word in enumerate(words):
            freq = self._word_counts[word.lower()]
            word_scores.append((freq, i, word))
            
        # Sort by frequency descending
        word_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Take top K
        process_indices = {x[1] for x in word_scores[:keep_count]}
        
        # Reconstruct list
        filtered_words = []
        for i, word in enumerate(words):
            if i in process_indices:
                filtered_words.append(word)
                
        return filtered_words

    def _update_dashboard_log(self):
        """Overwrites the log file with the current state (Dashboard)."""
        if self._log_file:
            try:
                # Format actions as a comma-separated string
                actions_str = ", ".join(self._last_actions) if self._last_actions else "None"
                
                content = (
                    f"[OSC STATUS]\n"
                    f"Counter      : {self._counter:.1f}\n"
                    f"Last Text    : {self._last_text}\n"
                    f"Last Actions : {actions_str}\n"
                )
                
                with self._log_file.open("w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                # Silently fail or minimal print to avoid spamming terminal
                pass

    def process_turn(self, text: str):
        """Called when the target persona speaks."""
        # Update stats first
        self._update_word_stats(text)
        
        # Filter text
        filtered_words_list = self._filter_text(text)
        filtered_str = " ".join(filtered_words_list)
        
        # Update State
        self._last_text = filtered_str
        self._last_actions = filtered_words_list
        
        # Send text
        # print(f"[OSC] Sending text: {filtered_str}", flush=True)
        self._client.send_message("/text", filtered_str)
        
        # Send retained words as "actions"
        for word in filtered_words_list:
             # print(f"[OSC] Sending action word: {word}", flush=True)
             self._client.send_message("/action", word) 
        
        # Update dashboard
        self._update_dashboard_log()

    def send_data(self):
        """Sends the numerical counter."""
        # Use formatted string to avoid float drift display issues, but send float type
        # print(f"[OSC] Sending data: {self._counter:.1f}", flush=True)
        self._client.send_message("/data", self._counter)
        
        # Update dashboard (counter changed)
        self._update_dashboard_log()

    def send_action(self, action: str):
        """Sends an extracted action. (Legacy manual trigger support)"""
        # print(f"[OSC] Sending action: {action}", flush=True)
        self._client.send_message("/action", action)
        
        # Treating legacy manual actions as part of the "Last Actions" list?
        # Or just appending? For dashboard, maybe just replace or append.
        # Let's replace for simplicity as it represents "Current Action"
        self._last_actions = [action]
        self._update_dashboard_log()
