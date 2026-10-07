const ML_URL = process.env.ML_API_URL || "http://127.0.0.1:8000";

export async function POST(request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");
    if (!file) {
      return Response.json({ error: "No file uploaded" }, { status: 400 });
    }

    const upstream = new FormData();
    upstream.append("file", file);

    const res = await fetch(`${ML_URL}/predict`, {
      method: "POST",
      body: upstream,
    });

    const data = await res.json();
    if (!res.ok) {
      return Response.json(
        { error: data.detail || "ML service error" },
        { status: res.status }
      );
    }

    return Response.json(data);
  } catch (err) {
    return Response.json(
      {
        error:
          "Cannot reach ML API. Start FastAPI: uvicorn api.main:app --reload --port 8000",
        detail: String(err.message),
      },
      { status: 503 }
    );
  }
}
