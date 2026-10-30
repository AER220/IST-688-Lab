# Lab6 - LangChain: the "What to Wear" bot, rebuilt
#
# this is the SAME bot as my Lab 5, but rebuilt with LangChain. the point of the lab is
# to see how much of my Lab 5 code LangChain writes for me. in Lab 5 i wrote, by hand:
#   - a JSON tool schema (the tools = [...] block)
#   - two separate API calls (one to detect the tool, one to answer)
#   - all the message bookkeeping in between (append the tool_call msg, run the tool,
#     append the tool result, send it all back)
# with LangChain, create_agent + agent.invoke do all of that. i only bring my weather
# function and my system prompt. everything that is NOT LangChain stays the same as Lab 5.

import os
import streamlit as st
from langchain.agents import create_agent

CHAT_MODEL = "openai:gpt-5-mini"   # same model as Lab 5, in LangChain's "provider:model" form

st.title("🧥 Lab 6 - What to Wear Bot (LangChain)")
st.write("Same bot as Lab 5, rebuilt with LangChain. Type a city and get advice.")

# LangChain's OpenAI integration reads the key from the OPENAI_API_KEY environment
# variable, not from OpenAI(api_key=...) like in Lab 5. so i copy my Streamlit secret there.
if "OPENAI_API_KEY" not in st.secrets:
    st.error("No OPENAI_API_KEY found in secrets. Add it to .streamlit/secrets.toml "
             "(and to your app's Secrets on Streamlit Cloud).")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# ------------------ Part B: my Lab 5 weather function, as a tool ------------------
# this is my get_weather_data from Lab 5, unchanged except for two things the lab asked for:
#   1. type hints on the signature
#   2. a docstring - this is NOT a comment. LangChain sends it to the model as the tool's
#      description, so the docstring is part of my prompt. i describe what it returns and
#      what kinds of locations it accepts so the model knows when and how to call it.
# i did NOT copy the Lab 5 tools = [...] JSON block or the json import - LangChain builds
# the schema from the function signature + docstring for me.
import requests

def get_weather_data(location: str = "Syracuse, NY") -> dict:
    """Get today's weather for a location, used to give clothing and activity advice.
    Accepts a city ('Syracuse, NY'), a landmark ('Eiffel Tower'), a zip code, or an
    airport code. Returns current temperature and feels-like, wind, humidity, today's
    high and low, the chance of rain, and a short description."""
    if not location:
        location = "Syracuse, NY"

    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return {"error": f"wttr.in error: status {response.status_code}"}
    try:
        data = response.json()
    except ValueError:
        return {"error": f"Could not find a location named {location}"}

    current = data["current_condition"][0]
    today = data["weather"][0]
    astronomy = today["astronomy"][0]
    max_rain = max(int(h["chanceofrain"]) for h in today["hourly"])

    area = data["nearest_area"][0]
    matched = ", ".join(
        part[0]["value"]
        for part in (area.get("areaName"), area.get("region"), area.get("country"))
        if part and part[0]["value"]
    )

    return {
        "location": matched or location,
        "current_temp_F": int(float(current["temp_F"])),
        "feels_like_F": int(float(current["FeelsLikeF"])),
        "description": current["weatherDesc"][0]["value"],
        "wind_mph": int(current["windspeedMiles"]),
        "humidity_pct": int(current["humidity"]),
        "today_high_F": int(today["maxtempF"]),
        "today_low_F": int(today["mintempF"]),
        "max_chance_of_rain": max_rain,
        "sunrise": astronomy["sunrise"],
        "sunset": astronomy["sunset"],
    }


# ------------------ Part C: build the agent ------------------
# same system prompt as Lab 5. in Lab 5 i passed this as the system message by hand;
# here i hand it to create_agent and LangChain manages the messages.
SYSTEM_PROMPT = (
    "You are a friendly 'what to wear' assistant. The user gives you a city. "
    "Use the get_weather_data tool to look up today's weather for that city, then "
    "recommend what clothes to wear today and suggest outdoor activities that suit the "
    "weather. Consider how conditions change through the day, the temperature and "
    "feels-like, wind, and the chance of rain. Keep it practical and easy to skim. "
    "If a city is not given, use Syracuse, NY."
)

# create_agent replaces the ENTIRE two-call loop i wrote in Lab 5. i give it the model,
# my function as a tool, and the system prompt - it builds the schema, decides when to
# call the tool, runs the loop, and returns the finished conversation.
agent = create_agent(
    model=CHAT_MODEL,
    tools=[get_weather_data],
    system_prompt=SYSTEM_PROMPT,
)

# the input + button from Lab 5
location = st.text_input("Enter a city", placeholder="Syracuse, NY")

if st.button("Get advice"):
    loc = location.strip() or "Syracuse, NY"

    # ONE call does everything: agent.invoke runs the whole tool loop internally.
    # in Lab 5 this was two client.chat.completions.create calls plus the bookkeeping between.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"What should I wear in {loc}?"}]}
    )

    # the answer is the LAST message in the returned conversation
    answer = result["messages"][-1].content
    with st.chat_message("assistant"):
        st.write(answer)

    # ------------------ Part D: look inside the agent ------------------
    # result["messages"] holds the WHOLE conversation, not just the answer.
    with st.expander("Behind the scenes"):
        for message in result["messages"]:
            st.write(f"**{message.type}**")
            st.write(message.content)
            if message.type == "ai" and message.tool_calls:
                st.write(message.tool_calls)


# ====================================================================
# Part D answers (step 12) - for one city like "Syracuse, NY":
#
# a. There are 4 messages:
#    1. human  - my question ("What should I wear in Syracuse, NY?")
#    2. ai     - the model's request to call get_weather_data (content is EMPTY here;
#                the request is in message.tool_calls, not in .content)
#    3. tool   - the weather dict my function returned
#    4. ai     - the final clothing + activity advice (this is result["messages"][-1])
#
# b. The model asks for the weather in message 2, the "ai" message. Its .content is empty;
#    the actual request (the function name + the location argument) lives in
#    message.tool_calls, which is why the Behind-the-scenes block prints that separately.
#
# c. In my Lab 5 code, I created those messages by hand:
#    - the human message: the {"role": "user", ...} dict I built from the city
#    - the ai tool-request: the first client.chat.completions.create call (tool_choice="auto"),
#      whose msg.tool_calls I then appended to my messages list
#    - the tool message: I ran get_weather_data myself and appended
#      {"role": "tool", "tool_call_id": ..., "content": ...}
#    - the final ai message: my SECOND client.chat.completions.create call
#    LangChain's single agent.invoke now creates all four of these for me.
# ====================================================================
