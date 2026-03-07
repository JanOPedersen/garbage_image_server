from marshmallow import Schema, fields


class PredictQuerySchema(Schema):
    model_id = fields.String(required=False)
    score_threshold = fields.Float(required=False, load_default=None)
    return_masks = fields.Boolean(required=False, load_default=False)


class DetectionSchema(Schema):
    label = fields.String(required=True)
    score = fields.Float(required=True)
    bbox_xyxy = fields.List(fields.Float(), required=True)
    mask_rle = fields.String(allow_none=True)


class PredictResponseSchema(Schema):
    request_id = fields.String(required=True)
    model_id = fields.String(required=True)
    image = fields.Dict(required=True)
    timing_ms = fields.Dict(required=True)
    detections = fields.List(fields.Nested(DetectionSchema), required=True)