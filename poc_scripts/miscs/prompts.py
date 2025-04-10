# The developer message is a prompt that is prepended to the user's input


developer_filename = "prompts/plan-generation-developer-message.txt"
with open(developer_filename, "r", encoding="utf-8") as file:
    DEVELOPER_MESSAGE = file.read()


executor_filename = "prompts/plan-generation-executor.txt"
with open(executor_filename, "r", encoding="utf-8") as file:
    EXECUTOR_MESSAGE = file.read()