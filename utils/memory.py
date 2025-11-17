class ConversationMemory:
    def __init__(self):
        self.history = []

    def add_user_message(self, msg):
        self.history.append({"role": "user", "content": msg})

    def add_assistant_message(self, msg):
        self.history.append({"role": "assistant", "content": msg})
