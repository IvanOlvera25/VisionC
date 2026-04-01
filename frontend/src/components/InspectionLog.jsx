"use client";

export default function InspectionLog({ log, mode }) {
  if (!log || log.length === 0) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>📋</span>
          <h3>Historial de Inspección</h3>
        </div>
        <div className="log-empty">
          <p>Sin inspecciones registradas</p>
        </div>
      </div>
    );
  }

  const isIndustrial = mode === "industrial";

  return (
    <div className="panel">
      <div className="panel-header">
        <span>📋</span>
        <h3>Historial de Inspección</h3>
        <span className="panel-count">{log.length}</span>
      </div>
      <div className="log-table-wrap">
        <table className="log-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Tipo</th>
              {isIndustrial ? (
                <>
                  <th>Diám.</th>
                  <th>Dientes</th>
                  <th>Defectos</th>
                </>
              ) : (
                <>
                  <th>Tamaño</th>
                  <th>Conf</th>
                </>
              )}
              <th>Estado</th>
            </tr>
          </thead>
          <tbody>
            {log.slice(0, 12).map((item, i) => (
              <tr key={i} className={`log-row ${item.ok ? "row-ok" : "row-fail"}`}>
                <td className="log-num">{i + 1}</td>
                <td className="log-class">
                  {isIndustrial ? (item.yolo_class || item.cls) : item.cls}
                </td>
                {isIndustrial ? (
                  <>
                    <td className="log-size">
                      {item.diameter ? (
                        <>Ø{item.diameter}<span className="unit">cm</span></>
                      ) : (
                        <>{item.w}×{item.h}<span className="unit">cm</span></>
                      )}
                    </td>
                    <td className="log-teeth">
                      <span className="teeth-badge">{item.teeth}T</span>
                      {!item.teeth_ok && (
                        <span className="defects-badge" style={{marginLeft: 4}}>≠</span>
                      )}
                    </td>
                    <td>
                      {item.defects && item.defects.length > 0 ? (
                        <span className="defects-badge">
                          {item.defects.length} ⚠️
                        </span>
                      ) : (
                        <span className="teeth-badge">0 ✓</span>
                      )}
                    </td>
                  </>
                ) : (
                  <>
                    <td className="log-size">
                      {item.w}×{item.h}
                      <span className="unit">cm</span>
                    </td>
                    <td className="log-conf">{(item.conf * 100).toFixed(0)}%</td>
                  </>
                )}
                <td>
                  <span className={`log-badge ${item.ok ? "badge-ok" : "badge-fail"}`}>
                    {item.ok ? "✅ OK" : "🔴 FAIL"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
