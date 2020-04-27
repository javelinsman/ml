import torch
import datetime
from sqlalchemy import Column, Integer, Unicode, DateTime, JSON
from .base import Base

class Tensor(Base):
    __tablename__ = 'tensors'
    
    id = Column(Integer, primary_key=True)
    step = Column(Integer, nullable=False)
    tag = Column(Unicode, nullable=False)
    tensor = Column(JSON, nullable=False)
    detail_dict = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f'Tensor(tag={self.tag}, step={self.step},' \
            f' {torch.tensor(self.tensor)}, {self.detail_dict})'

    def to_dict(self):
        return {
            'id': self.id,
            'step': self.step,
            'tag': self.tag,
            'tensor': self.tensor,
            'detail_dict': self.detail_dict,
            'timestamp': self.timestamp.timestamp()
        }