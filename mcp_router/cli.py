from __future__ import annotations
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--help-only", action="store_true")
    args = ap.parse_args()
    print("mcp_router package installed. Use scripts/ to run search/export/train.")