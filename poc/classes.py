from pydantic import BaseModel

class Message(BaseModel):
    type: str
    content: str


class MessageBlock(BaseModel):
    role: str
    items: list[Message]