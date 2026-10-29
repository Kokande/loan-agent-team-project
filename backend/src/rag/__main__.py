"""CLI загрузки и поиска: python -m src.rag --help."""

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import psycopg

from .config import create_rag
from .evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG по нормативным документам (pgvector)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Создать расширение и таблицы RAG")
    ingest = commands.add_parser("ingest", help="Загрузить документ или заменить его редакцию")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--source", required=True, help="Стабильный идентификатор документа")
    ingest.add_argument("--title")
    ingest.add_argument("--revision")
    ingest.add_argument("--effective-date", type=date.fromisoformat, help="YYYY-MM-DD")
    query = commands.add_parser("query", help="Найти релевантные чанки")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--min-score", type=float)
    delete = commands.add_parser("delete", help="Убрать документ и его чанки из корпуса")
    delete.add_argument("--source", required=True)
    evaluation = commands.add_parser("eval", help="Recall@k и MRR по JSONL")
    evaluation.add_argument("path", type=Path)
    evaluation.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    try:
        rag = create_rag()
        if args.command == "init-db":
            rag.store.initialize()
            result = {"initialized": True, "model": rag.embedder.model}
        elif args.command == "ingest":
            result = asdict(rag.ingest(
                args.path, args.source, args.title, args.revision, args.effective_date,
            ))
        elif args.command == "query":
            result = [asdict(chunk) for chunk in rag.retrieve(
                args.query, top_k=args.top_k, min_score=args.min_score,
            )]
        elif args.command == "delete":
            result = {"source": args.source, "deleted": rag.store.delete_document(args.source)}
        else:
            cases = [
                json.loads(line) for line in args.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            result = evaluate(cases, rag.retrieve, args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    except (ValueError, OSError, RuntimeError, psycopg.Error) as exc:
        parser.exit(1, f"Ошибка RAG: {exc}\n")


if __name__ == "__main__":
    main()
