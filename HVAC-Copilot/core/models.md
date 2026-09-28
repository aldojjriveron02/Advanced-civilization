# Initial Domain Models

## Equipment
- manufacturer
- model
- serial
- equipment_type
- identifiers_confidence

## Observation
- source: photo | voice | technician_input | document
- statement
- timestamp

## Measurement
- name
- value
- unit
- instrument
- timestamp
- quality

## Finding
- type: observed | calculated | manufacturer_fact | hypothesis | confirmed_repair
- statement
- supporting_measurements
- confidence

## DiagnosticStep
- question
- required_inputs
- calculation
- expected_ranges
- next_steps
- safety_notes
