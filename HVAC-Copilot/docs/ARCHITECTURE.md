# HVAC AI Copilot Architecture

## Core
The core owns HVAC-specific concepts: equipment, observations, measurements, diagnostic steps, findings, and service reports.

The core must not depend on iPhone UI or Meta hardware.

## Interfaces
The first interface is iPhone. Future wearable/AI interfaces call the same core API.

## Diagnostic safety
The system distinguishes observed facts, calculated values, manufacturer/documentation facts, hypotheses, and technician-confirmed repairs. An unverified hypothesis must never be presented as a confirmed diagnosis.
