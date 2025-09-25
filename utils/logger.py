#!/usr/bin/env python3
"""
Logging utility for Neural Galerkin TDD Workflow
Appends to a single consolidated .txt log file
"""

import os
import datetime
from pathlib import Path

class TDDLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "consolidated_tdd_log.txt"

    def get_readable_timestamp(self):
        """Get current timestamp in readable format"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def append_to_log(self, content):
        """Append content to the consolidated log file"""
        with open(self.log_file, "a") as f:
            f.write(content + "\n")

    def log_action(self, action, status="INFO", details=""):
        """Log an action to consolidated file"""
        readable_timestamp = self.get_readable_timestamp()

        # Status emoji mapping
        status_emojis = {
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "INFO": "📍",
            "IN_PROGRESS": "🔄",
            "COMPLETED": "✅"
        }

        emoji = status_emojis.get(status, "📝")

        # Create log entry
        log_entry = f"\n[{readable_timestamp}] {emoji} {status}: {action}"
        if details:
            log_entry += f"\nDETAILS: {details}"
        log_entry += f"\n{'-'*60}"

        # Append to consolidated log
        self.append_to_log(log_entry)
        print(f"[{readable_timestamp}] {emoji} {action}")

    def log_tdd_phase(self, phase, component, status, details=""):
        """Log TDD phase to consolidated file"""
        phase_emojis = {
            "RED": "🔴",
            "GREEN": "🟢",
            "VERIFY": "🔍"
        }

        emoji = phase_emojis.get(phase, "📝")
        readable_timestamp = self.get_readable_timestamp()

        log_entry = f"\n[{readable_timestamp}] {emoji} TDD {phase} PHASE: {component}"
        log_entry += f"\nSTATUS: {status}"
        if details:
            log_entry += f"\nDETAILS: {details}"
        log_entry += f"\n{'-'*60}"

        self.append_to_log(log_entry)
        print(f"[{readable_timestamp}] {emoji} TDD {phase}: {component} - {status}")

    def log_mathematical_result(self, test_name, result, error_value=None, tolerance=None):
        """Log mathematical test results to consolidated file"""
        readable_timestamp = self.get_readable_timestamp()

        if result == "PASS":
            emoji = "✅"
        else:
            emoji = "❌"

        log_entry = f"\n[{readable_timestamp}] {emoji} MATH TEST: {test_name}"
        log_entry += f"\nRESULT: {result}"

        if error_value is not None:
            log_entry += f"\nERROR: {error_value}"
        if tolerance is not None:
            log_entry += f"\nTOLERANCE: {tolerance}"
        log_entry += f"\n{'-'*60}"

        self.append_to_log(log_entry)
        print(f"[{readable_timestamp}] {emoji} Math Test: {test_name} - {result}")

    def log_code_execution(self, code_snippet, output, error=None):
        """Log code execution to consolidated file"""
        readable_timestamp = self.get_readable_timestamp()

        status_msg = "✅ SUCCESS" if error is None else "❌ FAILED"

        log_entry = f"\n[{readable_timestamp}] 💻 CODE EXECUTION"
        log_entry += f"\nCODE: {code_snippet}"
        if output:
            log_entry += f"\nOUTPUT: {output}"
        if error:
            log_entry += f"\nERROR: {error}"
        log_entry += f"\nSTATUS: {status_msg}"
        log_entry += f"\n{'-'*60}"

        self.append_to_log(log_entry)
        print(f"[{readable_timestamp}] 💻 Code: {code_snippet[:50]}... - {status_msg}")

# Convenience functions for easy logging
logger = TDDLogger()

def log_action(action, status="INFO", details=""):
    return logger.log_action(action, status, details)

def log_tdd_phase(phase, component, status, details=""):
    return logger.log_tdd_phase(phase, component, status, details)

def log_mathematical_result(test_name, result, error_value=None, tolerance=None):
    return logger.log_mathematical_result(test_name, result, error_value, tolerance)

def log_code_execution(code_snippet, output, error=None):
    return logger.log_code_execution(code_snippet, output, error)

if __name__ == "__main__":
    # Test the consolidated logger
    log_action("Consolidated Logger Updated", "SUCCESS", "Now appends to single consolidated_tdd_log.txt file instead of creating multiple files")