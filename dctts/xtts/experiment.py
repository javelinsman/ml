import os
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from pathlib import Path
import xtts.models as md


def mkdir(path, is_file):
    if is_file:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
    else:
        Path(path).mkdir(exist_ok=True, parents=True)

class Experiment:
    def __init__(self, name, base_dir=None):
        self.name = name
        self.base_dir = base_dir if base_dir else 'xtts_logs'
        mkdir(self.base_dir, is_file=False)
        self.log_dir = os.path.join(self.base_dir, self.name)
        mkdir(self.log_dir, is_file=False)
        self.__init_db()
        
    def __init_db(self):
        self.db_rel_path = os.path.join(self.log_dir, 'db.sqlite3')
        self.db = create_engine(f'sqlite:///{self.db_rel_path}')
        md.Base.metadata.create_all(self.db)
        self.Session = sessionmaker(bind=self.db)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
        
    def add_tensor(self, tag, step, tensor, detail_dict={}, timestamp=None):
        with self.session_scope() as session:
            tensor = md.Tensor(tag=tag, step=step,
                            tensor=tensor, detail_dict=detail_dict,
                            timestamp=timestamp)
            session.add(tensor)

    def fetch_tensors(self, tag='%', step='%'):
        with self.session_scope() as session:
            return session.query(md.Tensor) \
                .filter(md.Tensor.tag.like(tag)) \
                .filter(md.Tensor.step.like(step)).all()
