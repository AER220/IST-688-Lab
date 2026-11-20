# Lab 9 Agents - Multi-Agent Systems: the "What to Wear" crew.
#
# in Lab 5 i built the what to wear bot by hand (two api calls + my own tool loop),
# in Lab 6 i rebuilt it as ONE LangChain agent. here i split the same bot into a team
# with the OpenAI Agents SDK: a weather agent, a stylist and an activity planner,
# coordinated by a manager. then i compare the team with a single agent to see if the
# extra cost is worth it.
#
# my weather function is copied from Lab5.py unchanged, so the only thing that changes
# is how the work is divided.
#
# package vs import name: requirements.txt has openai-agents, but in code it is "agents".

import asyncio
import os
import time

import requests
import streamlit as st
from agents import Agent, ModelSettings, Runner, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

st.title("Lab 9 Agents - What to Wear Crew")
st.write("Type a city and a team of agents will tell you what to wear and do today.")

# same key check as Lab 5. the SDK reads the key from the environment variable.
if "OPENAI_API_KEY" not in st.secrets:
    st.error("No OPENAI_API_KEY found in secrets. Add it to .streamlit/secrets.toml "
             "(and to your app's Secrets on Streamlit Cloud).")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-5-mini"   # same model as my Lab 5 and Lab 6


# ------------------ Part B: my Lab 5 weather function (unchanged) ------------------
# get today's weather for a location from wttr.in.
# location can be a city, a zip code, an airport code like 'SYR', or a landmark.
# wttr.in is free and needs no api key. adding ?format=j1 makes it return json.
def get_current_weather(location):
    # if no location was given, default to Syracuse (matches step 6b)
    if not location:
        location = "Syracuse, NY"

    # build the url and ask wttr.in for json
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)

    # anything other than 200 means the request failed
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")

    # unknown places come back as plain text instead of json, so .json() will fail.
    # if that happens, treat it as "location not found".
    try:
        data = response.json()
    except ValueError:
        raise Exception(f"Could not find a location named {location}")

    # j1 has three sections i care about:
    #   current_condition -> conditions right now
    #   weather           -> one entry per day (i only use today, index 0)
    #   nearest_area      -> the place wttr.in actually matched my text to
    current = data["current_condition"][0]
    today = data["weather"][0]
    astronomy = today["astronomy"][0]

    # today["hourly"] is the forecast in 3-hour steps. i scan it for the day's
    # highest chance of rain and snow, so the bot knows if weather is coming later.
    max_rain = max(int(h["chanceofrain"]) for h in today["hourly"])
    max_snow = max(int(h["chanceofsnow"]) for h in today["hourly"])

    # build a small hour-by-hour list so the bot can see how the day changes.
    # wttr gives the time as 0, 300, 600 ... 2100 (basically HHMM with no colon),
    # so dividing by 100 gives the hour. e.g. 1500 -> 15 -> "15:00".
    hourly = []
    for h in today["hourly"]:
        hhmm = int(h["time"])
        hourly.append({
            "time": f"{hhmm // 100:02d}:00",
            "temp_F": int(h["tempF"]),
            "feels_like_F": int(h["FeelsLikeF"]),
            "chance_of_rain": int(h["chanceofrain"]),
            "description": h["weatherDesc"][0]["value"],
        })

    # build a readable name of the place wttr.in matched, e.g. "Syracuse, New York, United States".
    # each of these is a list with the value nested one level down, so i pull part[0]["value"].
    area = data["nearest_area"][0]
    matched = ", ".join(
        part[0]["value"]
        for part in (area.get("areaName"), area.get("region"), area.get("country"))
        if part and part[0]["value"]
    )

    # return everything as a clean dictionary. this is what gets handed to the model.
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
        "max_chance_of_snow": max_snow,
        "sunrise": astronomy["sunrise"],
        "sunset": astronomy["sunset"],
        "hourly": hourly,
    }




# the wrapper that turns my Lab 5 function into a tool. @function_tool replaces the
# function with a tool object that only an agent can call, so i keep the Lab 5 function
# separate and can still call it myself. the docstring and type hints become the tool
# description the model reads, same idea as Lab 6.
@function_tool
def get_weather(location: str = "Syracuse, NY") -> dict:
    """Get today's weather for one place. Returns a dictionary with the matched location
    name, current temperature and feels-like (F), description, wind (mph), humidity,
    today's high and low, the day's highest chance of rain and snow, sunrise and sunset,
    and an hourly outlook in 3-hour steps. Accepts a city ('Syracuse, NY', 'Lima, Peru'),
    a zip code, an airport code like 'SYR', or a landmark. If the place cannot be found
    it returns a dictionary with an 'error' key instead."""
    try:
        return get_current_weather(location)   # my Lab 5 function
    except Exception as e:
        # e.g. "purple banana" -> wttr.in has no such place. returning the error as data
        # lets the agents tell the user instead of the whole run crashing.
        return {"error": str(e), "location": location}


# ------------------ Part C: the team ------------------
weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "You are a weather reporter. Always call get_weather for the location you are given. "
        "Then summarise the facts in a few short lines: matched place, current temp and "
        "feels-like, today's high and low, wind, humidity, the chance of rain or snow, and how "
        "the day changes hour by hour. Give facts only, no clothing or activity advice. "
        "If the tool returns an error, say clearly that the location could not be found."
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You are a practical stylist. You are given a weather summary for today. Recommend "
        "what to wear: layers, top, bottoms, shoes and accessories (umbrella, hat, gloves, "
        "sunglasses), and say how to adjust through the day. Base every suggestion on the "
        "weather you were given. Keep it to a short bullet list. Do not suggest activities."
    ),
    model=MODEL,
)

activity_agent = Agent(
    name="Activity agent",
    instructions=(
        "You are an activity planner. You are given a weather summary for today. Suggest 3 "
        "to 4 activities that suit the weather, and say the best time of day for each, using "
        "the hourly outlook, sunrise and sunset. Prefer outdoor ideas when the weather allows "
        "and give indoor options when it is cold, wet or windy. Keep it short. Do not give "
        "clothing advice."
    ),
    model=MODEL,
)

manager_agent = Agent(
    name="Manager",
    instructions=(
        "You coordinate a what to wear team for one location. Step 1: always call "
        "weather_report first with the location. Step 2: send that weather summary to "
        "style_advice and to activity_ideas. Step 3: combine their answers into one friendly, "
        "easy to skim reply with a short weather line, a 'What to wear' section and a "
        "'What to do' section. Do not invent weather details yourself. If the weather report "
        "says the location could not be found, tell the user that, ask them to check the "
        "spelling, and do not call the other two tools. If no location is given, use "
        "Syracuse, NY."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description="Looks up today's weather for a location and returns a short "
                             "factual summary (no advice). Input: the location."),
        stylist_agent.as_tool(
            tool_name="style_advice",
            tool_description="Recommends what to wear today. Input: the weather summary "
                             "from weather_report."),
        activity_agent.as_tool(
            tool_name="activity_ideas",
            tool_description="Suggests activities that suit today's weather and when to do "
                             "them. Input: the weather summary from weather_report."),
    ],
    model=MODEL,
)


# ------------------ Part E: one agent doing everything ------------------
# same tool, one combined prompt (my Lab 6 system prompt, with the tool name updated).
SINGLE_PROMPT = (
    "You are a friendly 'what to wear' assistant. The user gives you a city. "
    "Use the get_weather tool to look up today's weather for that city, then "
    "recommend what clothes to wear today and suggest outdoor activities that suit the "
    "weather. Consider how conditions change through the day, the temperature and "
    "feels-like, wind, and the chance of rain. Keep it practical and easy to skim. "
    "If a city is not given, use Syracuse, NY. If the tool returns an error, tell the "
    "user the location could not be found."
)

single_agent = Agent(
    name="Single agent",
    instructions=SINGLE_PROMPT,
    tools=[get_weather],
    model=MODEL,
)


# ------------------ Bonus: triage with handoffs ------------------
# in a handoff the specialist takes over and answers the user itself,
# instead of reporting back to a manager.
packing_agent = Agent(
    name="Packing agent",
    handoff_description="Builds a packing list for a trip to another city.",
    instructions=prompt_with_handoff_instructions(
        "You have taken over from the triage agent and answer the user yourself. "
        "Always call get_weather for the destination first, then write a packing list."),
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="required"),   # first turn must be a tool call
    model=MODEL,
)

outfit_team = manager_agent.clone(name="Outfit team",
                                  handoff_description="What to wear and do today.")

triage_agent = Agent(
    name="Triage agent",
    instructions="Hand off trips to the Packing agent and "
                 "today's outfits to the Outfit team.",
    handoffs=[packing_agent, outfit_team],
    model=MODEL,
)


# ------------------ Part C step 2: run helper ------------------
# Runner.run_sync works on the first click, then fails with "Event loop is closed",
# because streamlit reruns the script on a new thread with a new event loop while the
# SDK's OpenAI client is still tied to the first one. one loop for the whole app fixes it.
@st.cache_resource
def get_event_loop():
    return asyncio.new_event_loop()


def run(agent, prompt):
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start


# ------------------ the page ------------------
mode = st.radio(
    "Who should answer?",
    ["Team (manager + 3 agents)", "Single agent", "Bonus: triage with handoffs"],
)

if mode == "Bonus: triage with handoffs":
    question = st.text_input(
        "Ask anything about today's outfit or a trip",
        value="I'm flying to Lima, Peru tomorrow for 3 days. What should I pack?",
    )
    clicked = st.button("Ask")
    agent, prompt = triage_agent, question
else:
    city = st.text_input("Enter a city", placeholder="Syracuse, NY")
    clicked = st.button("Get advice")
    location = city.strip() or "Syracuse, NY"
    agent = manager_agent if mode.startswith("Team") else single_agent
    prompt = f"What should I wear and do today in {location}?"

if clicked:
    with st.spinner("The agents are working (a team run can take 20 to 40 seconds)..."):
        try:
            result, seconds = run(agent, prompt)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    with st.chat_message("assistant"):
        st.write(result.final_output)

    # ------------------ Part D: behind the scenes ------------------
    usage = result.context_wrapper.usage
    with st.expander("Behind the scenes"):
        st.write(f"Model requests: {usage.requests} · Tokens: {usage.total_tokens:,} · "
                 f"Time: {seconds:.1f}s · Answered by: {result.last_agent.name}")
        for item in result.new_items:
            if item.type == "tool_call_item":
                st.write(f"{item.agent.name} called {item.raw_item.name}: "
                         f"{item.raw_item.arguments}")
            elif item.type == "tool_call_output_item":
                st.write(f"result: {str(item.output)[:300]}")
            elif item.type == "handoff_output_item":
                st.write(f"handoff: {item.source_agent.name} -> {item.target_agent.name}")


# ====================================================================
# Part D answers (step 2) - for one city like "Syracuse, NY":
#
# a. the manager called weather_report first, then style_advice and activity_ideas.
#    it never calls get_weather itself - only the weather agent has that tool. the manager
#    just gets back the weather agent's written report.
#
# b. a summary, not the full forecast. the raw dict from get_weather (hourly list and all)
#    stays inside the weather agent. what got sent to the stylist was the manager's own
#    short version of the weather - the manager model wrote that text when it decided
#    the arguments for style_advice (you can see it in the "called style_advice:" line).
#
# c. each agent-as-tool is a whole agent run on its own, and those inner calls don't show
#    up as items in the manager's list. the weather agent needs at least 2 requests (one
#    to ask for get_weather, one to write the summary), the stylist and activity agent need
#    1 each, and the manager needs its own (one to call the tools, one to write the final
#    answer). "Model requests" counts all of them, not just the ones i can see.
# ====================================================================
