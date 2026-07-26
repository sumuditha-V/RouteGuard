"""SQLite persistence for predictions + agent outputs (M8).

One table, `predictions`, with JSON columns for the flexible bits (SHAP drivers,
agent output, order snapshot). SQLite keeps the demo zero-setup; the same
SQLAlchemy code runs on Postgres by swapping the connection string (prod).
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "routeguard.sqlite"
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}",
                       connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    model_version: Mapped[str] = mapped_column(String)
    probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[str] = mapped_column(String)          # "late" | "on_time"
    threshold: Mapped[float] = mapped_column(Float)
    order: Mapped[dict] = mapped_column(JSON)                # readable order summary
    shap_drivers: Mapped[list] = mapped_column(JSON)
    agent: Mapped[dict] = mapped_column(JSON, nullable=True)  # None for /predict-only

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "model_version": self.model_version,
            "probability": self.probability,
            "prediction": self.prediction,
            "threshold": self.threshold,
            "order": self.order,
            "shap_drivers": self.shap_drivers,
            "agent": self.agent,
        }


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(engine)


def save_prediction(*, model_version, probability, prediction, threshold,
                    order, shap_drivers, agent=None) -> dict:
    with SessionLocal() as s:
        row = Prediction(
            created_at=datetime.now(timezone.utc),
            model_version=model_version, probability=probability,
            prediction=prediction, threshold=threshold, order=order,
            shap_drivers=shap_drivers, agent=agent,
        )
        s.add(row)
        s.commit()
        return row.as_dict()


def list_predictions(limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        rows = (s.query(Prediction).order_by(Prediction.id.desc())
                .limit(limit).all())
        return [r.as_dict() for r in rows]


def get_prediction(pred_id: int) -> dict | None:
    with SessionLocal() as s:
        row = s.get(Prediction, pred_id)
        return row.as_dict() if row else None
