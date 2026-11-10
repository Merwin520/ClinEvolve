from abc import ABC, abstractmethod

from agent.llm import OPENAI_MODEL
from utils.thread_logger import ThreadLoggerManager


class AgentSystem(ABC):
    def __init__(
        self,
        model=OPENAI_MODEL,
        chat_history_file="./outputs/chat_history.md",
        temperature=0.0,
        max_tokens=4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.logger_manager = ThreadLoggerManager(
            log_file=chat_history_file
        )

        self.log = self.logger_manager.log

        with open(
            chat_history_file,
            "w",
            encoding="utf-8",
        ) as f:
            f.write("")

    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ):
        pass