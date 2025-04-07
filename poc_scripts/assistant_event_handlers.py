from typing_extensions import override
from openai import AssistantEventHandler

class EventHandler(AssistantEventHandler):    
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)
        
    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
        
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
    
    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

# # Handle events in the stream
#             for event in stream:
#                 # print(f"[INFO] Event:\n {type(event)}")
#                 # if event == thread.run.step.delta:
#                 if isinstance(event, ThreadRunStepCreated):
#                     if event.data.step_details.type == "tool_calls":
#                         assistant_output.append({"type": "code_input",
#                                                 "content": ""})
                        
#                         code_input_expander = st.status("Writing code ...", expanded=True)
#                         code_input_block = code_input_expander.empty()

#                 if isinstance(event, ThreadRunStepDelta):
#                     if event.data.delta.step_details.tool_calls[0].code_interpreter is not None:
#                         code_interpreter = event.data.delta.step_details.tool_calls[0].code_interpreter
#                         code_input_delta = code_interpreter.input
#                         # print(f"[INFO] Code input delta: {code_input_delta}")
#                         if (code_input_delta is not None) and (code_input_delta != ""):
#                             assistant_output[-1]["content"] += code_input_delta
#                             code_input_block.empty()
#                             code_input_block.code(assistant_output[-1]["content"])
#                             # This part is added so that the 
#                         # code_input_expander.update(label="Code", state="complete", expanded=False)

#                 elif isinstance(event, ThreadRunStepCompleted):
#                     if isinstance(event.data.step_details, ToolCallsStepDetails):
#                         code_interpreter = event.data.step_details.tool_calls[0].code_interpreter
#                         code_input_expander.update(label="Code", state="complete", expanded=False)

#                         for output in code_interpreter.outputs:
#                             image_data_bytes_list = []
                            
#                             if isinstance(output, CodeInterpreterOutputImage):
#                                 image_file_id = output.image.file_id                                   
#                                 image_data = client.files.content(image_file_id)
#                                 image_data_bytes = image_data.read()
#                                 st.image(image_data_bytes)
#                                 image_data_bytes_list.append(image_data_bytes)

#                             if isinstance(output, CodeInterpreterOutputLogs):
#                                 # print(f"[INFO] This is a log. Show it in the code window.")
#                                 assistant_output.append({"type": "code_input",
#                                                         "content": ""})
#                                 code_output = output.logs
#                                 with st.status("Results", state="complete"):
#                                     st.code(code_output)
#                                     assistant_output[-1]["content"] = code_output

#                             assistant_output.append({
#                                 "type": "image",
#                                 "content":image_data_bytes_list
#                             })
                
#                 elif isinstance(event, ThreadMessageCreated):
#                     assistant_output.append({"type": "text",
#                                             "content": ""})
#                     assistant_text_box = st.empty()

#                 elif isinstance(event, ThreadMessageDelta):
#                     if isinstance(event.data.delta.content[0], TextDeltaBlock):
#                         assistant_text_box.empty()
#                         assistant_output[-1]["content"] += event.data.delta.content[0].text.value
#                         assistant_text_box.markdown(assistant_output[-1]["content"])

#             st.session_state.messages.append({"role": "assistant", "items": assistant_output})